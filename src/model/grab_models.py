"""GrabSingleTable and GrabMultiTable: LLM wrappers for precomputed GNN encoders."""

import contextlib
import json
import os
import re
import warnings

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .gnn_encoder import (
    GNNEncoderSplitConfig,
    GNNMultiTableEncoderSplitConfig,
    GNNTableEncoderPrecomputed,
    GNNMultiTableEncoderWithQuestionSplitPrecomputed,
)

warnings.filterwarnings("ignore", category=FutureWarning)

from src.model.lm_loss import causal_lm_loss_on_labels

IGNORE_INDEX = -100


def _build_llm(args):
    """Load LLM and apply frozen/LoRA/full-train setting. Returns the model."""
    llm_kwargs = {"device_map": {"": 0}, "revision": "main"}

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    if tokenizer.eos_token is None:
        if 'llama-3' in args.llm_model_name:
            tokenizer.eos_token = '<|end_of_text|>'
        elif 'qwen' in args.llm_model_name:
            tokenizer.eos_token = '<|endoftext|>'
        else:
            tokenizer.eos_token = '</s>'

    if tokenizer.bos_token is None:
        if 'llama-3' in args.llm_model_name:
            tokenizer.bos_token = '<|start_of_text|>'
        elif 'qwen' in args.llm_model_name:
            tokenizer.bos_token = '<|startoftext|>'
        else:
            tokenizer.bos_token = '<s>'

    if getattr(args, 'load_in_4bit', 'False') == 'True':
        llm_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    is_gemma4 = 'gemma4' in args.llm_model_name.lower() or 'gemma-4' in args.llm_model_name.lower()
    try:
        import flash_attn  # noqa: F401
        attn_impl = "sdpa" if is_gemma4 else "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        **llm_kwargs,
    )

    if args.llm_frozen == 'True':
        print("Freezing LLM parameters.")
        for param in model.parameters():
            param.requires_grad = False
        model.gradient_checkpointing_enable()
    elif args.llm_lora == 'True':
        print("Applying LoRA to LLM.")
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    else:
        print("LLM is fully trainable.")

    return tokenizer, model


def _build_projector(projector_type, encoder_dim, llm_dim, device):
    if projector_type == "mlp":
        proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
        )
    elif projector_type == "deep_mlp":
        proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim // 2),
            nn.GELU(),
            nn.LayerNorm(llm_dim // 2),
            nn.Linear(llm_dim // 2, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    else:
        proj = nn.Sequential(nn.Linear(encoder_dim, llm_dim))
    for module in proj.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return proj.to(device)


# ---------------------------------------------------------------------------
# GrabSingleTable (single-table, precomputed, instruct + base support)
# ---------------------------------------------------------------------------

class GrabSingleTable(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.dataset_name = args.dataset
        self.num_token = args.num_token
        self.args = args
        self.tokenizer, self.model = _build_llm(args)

        precomputed_dir = getattr(args, "precomputed_graphs", "")
        if not precomputed_dir:
            raise ValueError("--precomputed_graphs must be set for the precomputed model variant.")
        meta_path = os.path.join(precomputed_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json missing at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        hidden_size = int(meta["hidden_size"])
        self.precomputed_meta = meta

        num_row_latents = getattr(args, "num_row_latents", 4)
        num_col_latents = getattr(args, "num_col_latents", 2)
        num_val_latents = getattr(args, "num_val_latents", 2)

        gnn_config = GNNEncoderSplitConfig(
            base_model=getattr(args, "gnn_base_model", "sentence-transformers/all-MiniLM-L6-v2"),
            num_gnn_layers=getattr(args, "num_gnn_layers", 2),
            num_latents=num_row_latents + num_col_latents + num_val_latents,
            num_resampler_heads=getattr(args, "num_resampler_heads", 8),
            gnn_dropout=getattr(args, "gnn_dropout", 0.1),
            freeze_base_model=True,
            max_columns=getattr(args, "max_columns", 64),
            max_hash_groups=getattr(args, "max_hash_groups", 4096),
            max_header_len=getattr(args, "max_header_len", 32),
            num_row_latents=num_row_latents,
            num_col_latents=num_col_latents,
            num_val_latents=num_val_latents,
            num_resampler_layers=getattr(args, "num_resampler_layers", 1),
        )
        self.encoder_config = gnn_config

        self.table_encoder = GNNTableEncoderPrecomputed(config=gnn_config, hidden_size=hidden_size)
        self.table_encoder.to(self.model.device)

        if getattr(args, "table_encoder_frozen", "True") == "True":
            print("GNN table encoder is frozen.")
            for param in self.table_encoder.parameters():
                param.requires_grad = False
        else:
            print("GNN table encoder is unfrozen.")

        encoder_dim = self.table_encoder.hidden_size
        cfg = self.model.config
        llm_dim = getattr(cfg, 'hidden_size', None) or getattr(getattr(cfg, 'text_config', None), 'hidden_size', None)
        self.projector = _build_projector(
            getattr(args, "projector_type", "linear"), encoder_dim, llm_dim, self.model.device
        )

        self.is_instruct = 'Base' not in args.llm_model_path
        self.enable_thinking = getattr(args, 'enable_thinking', 'False') == 'True'
        self.no_question_conditioning = getattr(args, 'no_question_conditioning', 'False') == 'True'
        self.no_table_in_prompt = getattr(args, 'no_table_in_prompt', 'False') == 'True'

        self.word_embedding = self.model.model.get_input_embeddings()
        self._bos_embeds = None
        self._pad_embeds = None
        self._instruct_prefix_embeds = None
        self._instruct_user_suffix_ids = None

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

    def _get_chat_template_parts(self):
        _PLACEHOLDER = "__TABLE_ENCODER_PLACEHOLDER__"
        messages = [{"role": "user", "content": _PLACEHOLDER}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not self.enable_thinking:
            kwargs["enable_thinking"] = False
        try:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        prefix, suffix = text.split(_PLACEHOLDER)
        return prefix, suffix

    def _get_instruct_prefix_embeds(self):
        if self._instruct_prefix_embeds is None or self._instruct_prefix_embeds.device != self.device:
            prefix, _ = self._get_chat_template_parts()
            ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device)
            self._instruct_prefix_embeds = self.word_embedding(ids).detach()
        return self._instruct_prefix_embeds

    def _get_instruct_user_suffix_ids(self):
        if self._instruct_user_suffix_ids is None:
            _, suffix = self._get_chat_template_parts()
            self._instruct_user_suffix_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
        return self._instruct_user_suffix_ids

    @staticmethod
    def _strip_thinking(text):
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|channel>thought\n.*?<channel\|>\s*', '', text, flags=re.DOTALL)
        return text.strip()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.args, 'table_encoder_frozen', 'True') == 'True':
            if hasattr(self, 'table_encoder'):
                self.table_encoder.eval()
        return self

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
        g = samples["graph"]  # batched dict pre-padded by collate_graph_batch
        device = self.model.device

        R        = g["R"].to(device)
        C        = g["C"].to(device)
        row_mask = g["row_mask"].to(device)
        col_mask = g["col_mask"].to(device)
        adj      = g["adj"].to(device)
        adj_cv   = g["adj_cv"].to(device)
        g2c      = g["group_to_col"].to(device)
        q_tokens = g["q_tokens"].to(device) if "q_tokens" in g else None
        q_mask   = g["q_mask"].to(device)   if "q_mask"   in g else None
        vs       = g["value_stats"].to(device) if "value_stats" in g else None

        if self.no_question_conditioning:
            q_tokens = None
            q_mask   = None

        with torch.no_grad() if not self.training else contextlib.nullcontext():
            outputs = self.table_encoder(
                R=R, row_mask=row_mask, C=C, col_mask=col_mask,
                adj=adj, adj_cv=adj_cv, group_to_col=g2c,
                value_stats=vs, q_tokens=q_tokens, q_mask=q_mask,
            )
            return [outputs[i] for i in range(outputs.size(0))]

    def _prepare_table_embeddings(self, table_embeds_batch):
        processed = []
        for te in table_embeds_batch:
            if te.dim() == 1:
                te = te.unsqueeze(0)
            processed.append(self.projector(te))
        return processed

    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None and not self.no_table_in_prompt
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

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
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
                [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            )

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            return causal_lm_loss_on_labels(
                self.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids,
            )

    def inference(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None and not self.no_table_in_prompt
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

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
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
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = {
            'id': samples['id'],
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }
        result['pred'] = [self._strip_thinking(p) for p in pred]
        if self.enable_thinking:
            result['raw_pred'] = pred
        if 'question_type' in samples:
            result['question_type'] = samples['question_type']
        return result

    def print_trainable_params(self):
        def stats(module):
            if module is None:
                return 0, 0
            total = trainable = 0
            for p in module.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            return total, trainable

        enc_tot, enc_tr = stats(getattr(self, "table_encoder", None))
        llm_tot, llm_tr = stats(getattr(self, "model", None))
        prj_tot, prj_tr = stats(getattr(self, "projector", None))
        full_tot, full_tr = stats(self)
        known_tot = enc_tot + llm_tot + prj_tot
        known_tr = enc_tr + llm_tr + prj_tr
        oth_tot = max(0, full_tot - known_tot)
        oth_tr = max(0, full_tr - known_tr)

        def pct(tr, tot):
            return 100.0 * tr / tot if tot > 0 else 0.0

        print("\n" + "=" * 92)
        print(f"| {'COMPONENT':<22} | {'TOTAL PARAMS':>15} | {'TRAINABLE':>15} | {'% TRAINABLE':>11} |")
        print("|" + "-" * 90 + "|")
        for name, tot, tr in [
            ("Table Encoder", enc_tot, enc_tr),
            ("LLM", llm_tot, llm_tr),
            ("Projector", prj_tot, prj_tr),
            ("Other", oth_tot, oth_tr),
            ("TOTAL", full_tot, full_tr),
        ]:
            print(f"| {name:<22} | {tot:>15,} | {tr:>15,} | {pct(tr, tot):>10.2f}% |")
        print("=" * 92)
        return full_tr, full_tot


# ---------------------------------------------------------------------------
# GrabMultiTable (multi-table, precomputed, base model only)
# ---------------------------------------------------------------------------

class GrabMultiTable(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.dataset_name = args.dataset
        self.num_token = args.num_token
        self.args = args
        self.no_question_conditioning = getattr(args, 'no_question_conditioning', 'False') == 'True'
        self.no_table_in_prompt = getattr(args, 'no_table_in_prompt', 'False') == 'True'
        self.tokenizer, self.model = _build_llm(args)

        precomputed_dir = getattr(args, 'precomputed_graphs', '')
        if not precomputed_dir:
            raise ValueError("--precomputed_graphs must be set for the precomputed model variant.")
        meta_path = os.path.join(precomputed_dir, 'meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json missing at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        hidden_size = int(meta['hidden_size'])
        self.precomputed_meta = meta

        num_row_latents = getattr(args, 'num_row_latents', 4)
        num_col_latents = getattr(args, 'num_col_latents', 2)
        num_val_latents = getattr(args, 'num_val_latents', 2)

        gnn_config = GNNMultiTableEncoderSplitConfig(
            base_model=getattr(args, 'gnn_base_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            num_gnn_layers=getattr(args, 'num_gnn_layers', 2),
            num_latents=num_row_latents + num_col_latents + num_val_latents,
            num_resampler_heads=getattr(args, 'num_resampler_heads', 8),
            num_resampler_layers=getattr(args, 'num_resampler_layers', 1),
            gnn_dropout=getattr(args, 'gnn_dropout', 0.1),
            freeze_base_model=True,
            max_columns=getattr(args, 'max_columns', 64),
            max_hash_groups=getattr(args, 'max_hash_groups', 4096),
            max_header_len=getattr(args, 'max_header_len', 32),
            num_row_latents=num_row_latents,
            num_col_latents=num_col_latents,
            num_val_latents=num_val_latents,
            max_tables=getattr(args, 'max_tables', 8),
        )
        self.encoder_config = gnn_config

        self.table_encoder = GNNMultiTableEncoderWithQuestionSplitPrecomputed(
            config=gnn_config, hidden_size=hidden_size,
        )
        self.table_encoder.to(self.model.device)

        if getattr(args, 'table_encoder_frozen', 'True') == 'True':
            print("GNN multi-table encoder is frozen.")
            for param in self.table_encoder.parameters():
                param.requires_grad = False
        else:
            print("GNN multi-table encoder is unfrozen.")

        encoder_dim = self.table_encoder.hidden_size
        cfg = self.model.config
        llm_dim = getattr(cfg, 'hidden_size', None) or getattr(getattr(cfg, 'text_config', None), 'hidden_size', None)
        self.projector = _build_projector(
            getattr(args, 'projector_type', 'linear'), encoder_dim, llm_dim, self.model.device
        )

        self.word_embedding = self.model.model.get_input_embeddings()
        self._bos_embeds = None
        self._pad_embeds = None

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

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.args, 'table_encoder_frozen', 'True') == 'True':
            if hasattr(self, 'table_encoder'):
                self.table_encoder.eval()
        return self

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
        g = samples['graph']  # batched dict pre-padded by collate_graph_batch
        device = self.model.device

        R        = g['R'].to(device)
        C        = g['C'].to(device)
        row_mask = g['row_mask'].to(device)
        col_mask = g['col_mask'].to(device)
        adj      = g['adj'].to(device)
        adj_cv   = g['adj_cv'].to(device)
        g2c      = g['group_to_col'].to(device)
        q_tokens = g['q_tokens'].to(device)    if 'q_tokens'      in g else None
        q_mask   = g['q_mask'].to(device)      if 'q_mask'        in g else None
        vs       = g['value_stats'].to(device) if 'value_stats'   in g else None
        row_tids = g['row_table_ids'].to(device) if 'row_table_ids' in g else None
        col_tids = g['col_table_ids'].to(device) if 'col_table_ids' in g else None
        num_tables_t = g['num_tables'].to(device) if 'num_tables'   in g else None

        if self.no_question_conditioning:
            q_tokens = None
            q_mask   = None

        with torch.no_grad() if not self.training else contextlib.nullcontext():
            outputs = self.table_encoder(
                R=R, row_mask=row_mask, C=C, col_mask=col_mask,
                adj=adj, adj_cv=adj_cv, group_to_col=g2c,
                value_stats=vs, q_tokens=q_tokens, q_mask=q_mask,
                row_table_ids=row_tids, col_table_ids=col_tids,
                num_tables=num_tables_t,
            )
            return [outputs[i] for i in range(outputs.size(0))]

    def _prepare_table_embeddings(self, table_embeds_batch):
        processed = []
        for te in table_embeds_batch:
            if te.dim() == 1:
                te = te.unsqueeze(0)
            processed.append(self.projector(te))
        return processed

    def _build_text_input_ids(self, samples, idx, desc_ids, question_ids, extra_ids=None):
        if extra_ids is not None:
            seg_ids = []
            for seg_toks in extra_ids:
                seg_ids += seg_toks[:self.max_txt_len]
            return desc_ids[:self.max_txt_len] + seg_ids + question_ids
        else:
            return desc_ids[:self.max_txt_len] + question_ids

    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None and not self.no_table_in_prompt
        if has_segs:
            table_segs_tokens = [
                self.tokenizer(segs, add_special_tokens=False).input_ids
                for segs in samples['table_segs']
            ]

        table_embeds_batch = self.encode_tables(samples)
        table_embeds_batch = self._prepare_table_embeddings(table_embeds_batch)

        eos_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        bos_embeds = self._get_bos_embeds()
        pad_embeds = self._get_pad_embeds()

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            text_ids = self._build_text_input_ids(
                samples, i,
                descriptions.input_ids[i],
                questions.input_ids[i],
                extra_ids=table_segs_tokens[i] if has_segs else None,
            ) + label_input_ids

            inputs_embeds = self.word_embedding(torch.as_tensor(text_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, table_embeds_batch[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            batch_label_input_ids.append(
                [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            )

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            return causal_lm_loss_on_labels(
                self.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids,
            )

    def inference(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        has_segs = 'table_segs' in samples and samples['table_segs'] is not None and not self.no_table_in_prompt
        if has_segs:
            table_segs_tokens = [
                self.tokenizer(segs, add_special_tokens=False).input_ids
                for segs in samples['table_segs']
            ]

        bos_embeds = self._get_bos_embeds()
        pad_embeds = self._get_pad_embeds()

        table_embeds_batch = self.encode_tables(samples)
        table_embeds_batch = self._prepare_table_embeddings(table_embeds_batch)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            text_ids = self._build_text_input_ids(
                samples, i,
                descriptions.input_ids[i],
                questions.input_ids[i],
                extra_ids=table_segs_tokens[i] if has_segs else None,
            )
            inputs_embeds = self.word_embedding(torch.as_tensor(text_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, table_embeds_batch[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device, dtype=torch.bfloat16)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        result = {
            'id': samples['id'],
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }
        if 'eval_meta' in samples:
            result['eval_meta'] = samples['eval_meta']
        if 'question_type' in samples:
            result['question_type'] = samples['question_type']
        return result

    def print_trainable_params(self):
        def stats(module):
            if module is None:
                return 0, 0
            total = trainable = 0
            for p in module.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            return total, trainable

        enc_tot, enc_tr = stats(getattr(self, "table_encoder", None))
        llm_tot, llm_tr = stats(getattr(self, "model", None))
        prj_tot, prj_tr = stats(getattr(self, "projector", None))
        full_tot, full_tr = stats(self)

        def pct(tr, tot):
            return 100.0 * tr / tot if tot > 0 else 0.0

        print("\n" + "=" * 92)
        print(f"| {'COMPONENT':<22} | {'TOTAL PARAMS':>15} | {'TRAINABLE':>15} | {'% TRAINABLE':>11} |")
        print("|" + "-" * 90 + "|")
        for name, tot, tr in [
            ("GNN Multi Encoder", enc_tot, enc_tr),
            ("LLM", llm_tot, llm_tr),
            ("Projector", prj_tot, prj_tr),
            ("FULL MODEL", full_tot, full_tr),
        ]:
            print(f"| {name:<22} | {tot:>15,} | {tr:>15,} | {pct(tr, tot):>10.2f}% |")
        print("=" * 92 + "\n")
        return full_tr, full_tot
