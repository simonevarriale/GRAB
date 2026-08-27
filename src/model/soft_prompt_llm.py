"""LLM wrapper for the soft-prompt-only table encoder.

Identical interface to the GNN models but replaces the entire GNN encoder +
tokenizer with a simple learned soft prompt. The table content is completely
ignored — only the number of samples in the batch matters.
"""

import contextlib
import re
import warnings
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

from src.model.lm_loss import causal_lm_loss_on_labels

IGNORE_INDEX = -100

@dataclass
class SoftPromptEncoderConfig:
    hidden_size: int = 384
    num_latents: int = 64
    num_mlp_layers: int = 0  # 0 = no MLP (raw soft prompt), >0 = MLP on top

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SoftPromptTableEncoder(nn.Module):
    """Returns `num_latents` learnable vectors, independent of the input table."""

    def __init__(self, config=None, *, hidden_size=384, num_latents=64, num_mlp_layers=0):
        super().__init__()

        if config is not None:
            if not isinstance(config, SoftPromptEncoderConfig):
                raise TypeError(f"Expected SoftPromptEncoderConfig, got {type(config)}")
            self.config = config
        else:
            self.config = SoftPromptEncoderConfig(
                hidden_size=hidden_size,
                num_latents=num_latents,
                num_mlp_layers=num_mlp_layers,
            )

        self.hidden_size = self.config.hidden_size
        self.latents = nn.Parameter(
            torch.randn(1, self.config.num_latents, self.config.hidden_size) * 0.02
        )

        if self.config.num_mlp_layers > 0:
            layers = []
            for _ in range(self.config.num_mlp_layers):
                layers.extend([
                    nn.LayerNorm(self.config.hidden_size),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
                ])
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None

    def forward(self, batch_size: int):
        """Returns [B, num_latents, hidden_size]."""
        out = self.latents.expand(batch_size, -1, -1)
        if self.mlp is not None:
            out = out + self.mlp(out)
        return out



class TableSoftPromptLLM(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len    = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.dataset_name   = args.dataset
        self.num_token      = args.num_token
        self.args           = args

        # --- LLM ---
        llm_kwargs = {"device_map": {"": 0}, "revision": "main"}

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.eos_token is None:
            if 'llama-3' in args.llm_model_name:
                self.tokenizer.eos_token = '<|end_of_text|>'
            elif 'qwen' in args.llm_model_name:
                self.tokenizer.eos_token = '<|endoftext|>'
            else:
                self.tokenizer.eos_token = '</s>'

        if self.tokenizer.bos_token is None:
            if 'llama-3' in args.llm_model_name:
                self.tokenizer.bos_token = '<|start_of_text|>'
            elif 'qwen' in args.llm_model_name:
                self.tokenizer.bos_token = '<|startoftext|>'
            else:
                self.tokenizer.bos_token = '<s>'


        is_gemma4 = 'gemma4' in args.llm_model_name.lower() or 'gemma-4' in args.llm_model_name.lower()
        try:
            import flash_attn  # noqa: F401
            attn_impl = "sdpa" if is_gemma4 else "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"Using attention implementation: {attn_impl}")

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            **llm_kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLM parameters.")
            for param in model.parameters(): param.requires_grad = False
            model.gradient_checkpointing_enable()
        elif args.llm_lora == 'True':
            print("Applying LoRA to LLM.")
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        else:
            print("LLM is fully trainable.")

        self.model = model

        # --- Soft-prompt encoder ---
        hidden_size    = getattr(args, 'gnn_hidden_size', 384)
        num_latents    = getattr(args, 'num_latents', 64)
        num_mlp_layers = getattr(args, 'num_mlp_layers', 0)

        enc_config = SoftPromptEncoderConfig(
            hidden_size=hidden_size,
            num_latents=num_latents,
            num_mlp_layers=num_mlp_layers,
        )
        self.encoder_config = enc_config
        self.table_encoder = SoftPromptTableEncoder(config=enc_config)
        self.table_encoder.to(self.model.device)

        # --- Projector ---
        encoder_dim    = self.table_encoder.hidden_size
        llm_dim        = (lambda c: getattr(c, "hidden_size", None) or getattr(getattr(c, "text_config", None), "hidden_size", None))(self.model.config)
        projector_type = getattr(args, 'projector_type', 'linear')

        if projector_type == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(encoder_dim, llm_dim // 2),
                nn.GELU(),
                nn.Linear(llm_dim // 2, llm_dim),
            ).to(self.model.device)
        elif projector_type == 'deep_mlp':
            self.projector = nn.Sequential(
                nn.Linear(encoder_dim, llm_dim // 2),
                nn.GELU(),
                nn.LayerNorm(llm_dim // 2),
                nn.Linear(llm_dim // 2, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim),
            ).to(self.model.device)
        else:
            self.projector = nn.Sequential(
                nn.Linear(encoder_dim, llm_dim),
            ).to(self.model.device)

        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.word_embedding = self.model.model.get_input_embeddings()
        self._bos_embeds = None
        self._pad_embeds = None
        self._instruct_prefix_embeds = None
        self._instruct_user_suffix_ids = None

        self.is_instruct = not (
            'base' in args.llm_model_name.lower() or
            'base' in args.llm_model_path.lower()
        )
        self.enable_thinking = getattr(args, 'enable_thinking', 'False') == 'True'

    def _get_bos_embeds(self):
        if self._bos_embeds is None or self._bos_embeds.device != self.device:
            bos_ids = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt'
            ).input_ids[0].to(self.device)
            self._bos_embeds = self.word_embedding(bos_ids).detach()
        return self._bos_embeds

    def _get_pad_embeds(self):
        if self._pad_embeds is None or self._pad_embeds.device != self.device:
            self._pad_embeds = self.word_embedding(
                torch.tensor(self.tokenizer.pad_token_id, device=self.device)
            ).unsqueeze(0).detach()
        return self._pad_embeds

    def _get_instruct_prefix_embeds(self):
        if self._instruct_prefix_embeds is None or self._instruct_prefix_embeds.device != self.device:
            text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device)
            self._instruct_prefix_embeds = self.word_embedding(ids).detach()
        return self._instruct_prefix_embeds

    def _get_instruct_user_suffix_ids(self):
        if self._instruct_user_suffix_ids is None:
            if self.enable_thinking:
                text = "<|im_end|>\n<|im_start|>assistant\n"
            else:
                text = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
            self._instruct_user_suffix_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return self._instruct_user_suffix_ids

    @staticmethod
    def _strip_thinking(text):
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    @classmethod
    def from_pretrained(cls, checkpoint_path, args=None):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if args is None:
            args = checkpoint["config"]
        args.do_eval = 'True'
        instance = cls(args)
        instance.load_state_dict(checkpoint["model"], strict=False)
        instance.eval()
        return instance

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.device.type != 'cpu' else contextlib.nullcontext()

    def encode_tables(self, samples):
        """Ignores table content — returns the same soft prompt for every sample."""
        batch_size = len(samples['table'])
        outputs = self.table_encoder(batch_size)  # [B, num_latents, H]
        return [outputs[i] for i in range(batch_size)]

    def _prepare_table_embeddings(self, table_embeds_batch):
        processed_batch = []
        for table_embeds in table_embeds_batch:
            if table_embeds.dim() == 1:
                table_embeds = table_embeds.unsqueeze(0)
            processed_batch.append(self.projector(table_embeds))
        return processed_batch

    def forward(self, samples):
        questions    = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"],     add_special_tokens=False)
        labels       = self.tokenizer(samples["label"],    add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None
        if has_segs:
            table_segs_tokens = [
                self.tokenizer(segs, add_special_tokens=False).input_ids
                for segs in samples['table_segs']
            ]

        table_embeds_batch = self.encode_tables(samples)
        table_embeds_batch = self._prepare_table_embeddings(table_embeds_batch)

        eos_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        pad_embeds = self._get_pad_embeds()
        if not self.is_instruct:
            bos_embeds = self._get_bos_embeds()

        batch_size = len(samples['id'])
        batch_inputs_embeds   = []
        batch_attention_mask  = []
        batch_label_input_ids = []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            if has_segs:
                seg_ids = []
                for seg_toks in table_segs_tokens[i]:
                    seg_ids += seg_toks[:self.max_txt_len]
                desc_q_ids = descriptions.input_ids[i] + seg_ids + questions.input_ids[i]
            else:
                desc_q_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i]

            if self.is_instruct:
                suffix_ids = self._get_instruct_user_suffix_ids()
                text_ids = desc_q_ids + suffix_ids + label_input_ids
                text_embeds = self.word_embedding(torch.as_tensor(text_ids, device=self.device))
                inputs_embeds = torch.cat([self._get_instruct_prefix_embeds(), table_embeds_batch[i], text_embeds], dim=0)
            else:
                text_ids = desc_q_ids + label_input_ids
                text_embeds = self.word_embedding(torch.as_tensor(text_ids, device=self.device))
                inputs_embeds = torch.cat([bos_embeds, table_embeds_batch[i], text_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            batch_label_input_ids.append(
                [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids))
                + label_input_ids
            )

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i]   = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i]  = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds   = torch.stack(batch_inputs_embeds,   dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask  = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            return causal_lm_loss_on_labels(
                self.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids,
            )

    def inference(self, samples):
        questions    = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"],     add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None
        if has_segs:
            table_segs_tokens = [
                self.tokenizer(segs, add_special_tokens=False).input_ids
                for segs in samples['table_segs']
            ]

        pad_embeds = self._get_pad_embeds()
        if not self.is_instruct:
            bos_embeds = self._get_bos_embeds()

        table_embeds_batch = self.encode_tables(samples)
        table_embeds_batch = self._prepare_table_embeddings(table_embeds_batch)

        batch_size = len(samples['id'])
        batch_inputs_embeds  = []
        batch_attention_mask = []

        for i in range(batch_size):
            if has_segs:
                seg_ids = []
                for seg_toks in table_segs_tokens[i]:
                    seg_ids += seg_toks[:self.max_txt_len]
                desc_q_ids = descriptions.input_ids[i] + seg_ids + questions.input_ids[i]
            else:
                desc_q_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i]

            if self.is_instruct:
                suffix_ids = self._get_instruct_user_suffix_ids()
                text_ids = desc_q_ids + suffix_ids
                text_embeds = self.word_embedding(torch.as_tensor(text_ids, device=self.device))
                inputs_embeds = torch.cat([self._get_instruct_prefix_embeds(), table_embeds_batch[i], text_embeds], dim=0)
            else:
                text_embeds = self.word_embedding(torch.as_tensor(desc_q_ids, device=self.device))
                inputs_embeds = torch.cat([bos_embeds, table_embeds_batch[i], text_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i]  = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds  = torch.stack(batch_inputs_embeds,  dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        result = {
            'id':       samples['id'],
            'pred':     pred,
            'label':    samples['label'],
            'question': samples['question'],
            'desc':     samples['desc'],
        }
        if self.enable_thinking:
            result['raw_pred'] = pred
            result['pred'] = [self._strip_thinking(p) for p in pred]
        if 'eval_meta' in samples:
            result['eval_meta'] = samples['eval_meta']
        if 'question_type' in samples:
            result['question_type'] = samples['question_type']
        return result

    def print_trainable_params(self):
        def stats(module):
            if module is None: return 0, 0
            total = trainable = 0
            for p in module.parameters():
                total += p.numel()
                if p.requires_grad: trainable += p.numel()
            return total, trainable

        enc_tot, enc_tr = stats(getattr(self, "table_encoder", None))
        llm_tot, llm_tr = stats(getattr(self, "model", None))
        prj_tot, prj_tr = stats(getattr(self, "projector", None))

        full_tot, full_tr = stats(self)
        known_tot = enc_tot + llm_tot + prj_tot
        known_tr  = enc_tr  + llm_tr  + prj_tr
        oth_tot   = max(0, full_tot - known_tot)
        oth_tr    = max(0, full_tr  - known_tr)

        def pct(tr, tot): return 100.0 * tr / tot if tot > 0 else 0.0

        print("\n" + "=" * 92)
        print(f"| {'COMPONENT':<22} | {'TOTAL PARAMS':>15} | {'TRAINABLE':>15} | {'% TRAINABLE':>11} |")
        print("|" + "-" * 90 + "|")
        if enc_tot: print(f"| {'Soft Prompt Encoder':<22} | {enc_tot:>15,} | {enc_tr:>15,} | {pct(enc_tr, enc_tot):>10.2f}% |")
        if llm_tot: print(f"| {'LLM':<22} | {llm_tot:>15,} | {llm_tr:>15,} | {pct(llm_tr, llm_tot):>10.2f}% |")
        if prj_tot: print(f"| {'Projector':<22} | {prj_tot:>15,} | {prj_tr:>15,} | {pct(prj_tr, prj_tot):>10.2f}% |")
        if oth_tot: print(f"| {'Other':<22} | {oth_tot:>15,} | {oth_tr:>15,} | {pct(oth_tr, oth_tot):>10.2f}% |")
        print("|" + "-" * 90 + "|")
        print(f"| {'FULL MODEL':<22} | {full_tot:>15,} | {full_tr:>15,} | {pct(full_tr, full_tot):>10.2f}% |")
        print("=" * 92 + "\n")
        return full_tr, full_tot
