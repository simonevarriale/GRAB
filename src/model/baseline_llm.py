import os
import json
import re
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

from src.model.lm_loss import causal_lm_loss_on_labels

IGNORE_INDEX = -100
fetaqa_question_len = 512


class BaselineLLM(torch.nn.Module):
    """
    Baseline LLM without custom table encoder.
    Tables are serialized to text and processed directly by the LLM.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.dataset_name = args.dataset
        self.args = args

        # Table serialization format
        self.table_serialization = getattr(args, 'table_serialization', 'markdown')  # 'markdown', 'json', 'csv', 'linearized'

        kwargs = {"device_map": "auto", "revision": "main"}
        if args.llm_frozen == 'False' and args.llm_lora == 'False' and args.do_eval == 'False':
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                kwargs["max_memory"] = {i: '80GiB' for i in range(num_gpus)}
            elif num_gpus == 1:
                kwargs["max_memory"] = {0: '80GiB'}

        # nf4 4-bit quantization (bitsandbytes): shrinks a frozen backbone ~4x so a
        # 70B fits on a single GPU. device_map="auto" places the quantized model on
        # the rank's visible GPU.
        if getattr(args, 'load_in_4bit', 'False') == 'True':
            from transformers import BitsAndBytesConfig
            print("Loading LLM in nf4 4-bit (bitsandbytes).")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # Force the whole quantized model onto the rank's single GPU. "auto"
            # applies a headroom heuristic and offloads layers to CPU, which bnb
            # quantization rejects ("Some modules are dispatched on the CPU or the disk").
            kwargs["device_map"] = {"": 0}
        elif getattr(args, 'load_in_8bit', 'False') == 'True':
            from transformers import BitsAndBytesConfig
            print("Loading LLM in int8 8-bit (bitsandbytes).")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": 0}

        use_fast = (
            'llama-3' in args.llm_model_name or
            'llama3' in args.llm_model_name or
            'gemma4' in args.llm_model_name or
            'gemma-4' in args.llm_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=use_fast, trust_remote_code=True)
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

        print(f"bos_token: {self.tokenizer.bos_token}")

        # A model with no chat template cannot be instruct-tuned, regardless of name.
        # This guards base models (e.g. Llama-3.1-70B) whose name lacks "base" from
        # wrongly taking the chat-template inference path.
        self.is_instruct = not (
            'base' in args.llm_model_name.lower() or
            'base' in args.llm_model_path.lower()
        ) and getattr(self.tokenizer, 'chat_template', None) is not None
        self.enable_thinking = getattr(args, 'enable_thinking', 'False') == 'True'
        if self.is_instruct:
            print("Instruct model detected: will use chat template for inference/training.")
        if self.enable_thinking:
            print("Thinking/reasoning enabled: <think> blocks will be stripped from predictions.")

        is_gemma4 = 'gemma4' in args.llm_model_name.lower() or 'gemma-4' in args.llm_model_name.lower()
        try:
            import flash_attn  # noqa: F401
            attn_impl = "sdpa" if is_gemma4 else "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"Using attention implementation: {attn_impl}")

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLM parameters.")
            for param in model.parameters():
                param.requires_grad = False
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
        self.word_embedding = self.model.model.get_input_embeddings()

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
        return torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.device.type != 'cpu' else contextlib.nullcontext()

    def serialize_table(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df = df.astype(str)

        if self.table_serialization == 'markdown':
            return df.to_markdown(index=False)
        elif self.table_serialization == 'csv':
            return df.to_csv(index=False)
        elif self.table_serialization == 'json':
            return df.to_json(orient='records')
        elif self.table_serialization == 'linearized':
            rows = []
            for _, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                rows.append(row_str)
            return "\n".join(rows)
        else:
            rows = []
            for _, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                rows.append(row_str)
            return "\n".join(rows)

    def _build_desc_ids(self, samples, idx, descriptions):
        table_descs = samples.get('table_descs') or samples.get('table_segs')
        if table_descs is not None and table_descs[idx] is not None:
            per_table_ids = []
            for html in table_descs[idx]:
                tok = self.tokenizer(html, add_special_tokens=False)
                per_table_ids.extend(tok.input_ids[:self.max_txt_len])

            full_desc_ids = descriptions.input_ids[idx]
            marker = self.tokenizer("### Input:\n", add_special_tokens=False).input_ids
            marker_len = len(marker)
            prefix_end = None
            for pos in range(len(full_desc_ids) - marker_len + 1):
                if full_desc_ids[pos:pos + marker_len] == marker:
                    prefix_end = pos + marker_len
                    break
            if prefix_end is not None:
                return full_desc_ids[:prefix_end] + per_table_ids
            else:
                return per_table_ids
        else:
            return descriptions.input_ids[idx][:self.max_txt_len]

    def forward(self, samples):
        if self.is_instruct:
            return self._forward_instruct(samples)

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        eos_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            .input_ids[0].to(self.device)
        )
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            desc_ids = self._build_desc_ids(samples, i, descriptions)

            if self.dataset_name == 'fetaqa':
                input_ids = (desc_ids +
                           questions.input_ids[i][:fetaqa_question_len] +
                           label_input_ids)
            else:
                input_ids = (desc_ids +
                           questions.input_ids[i] +
                           label_input_ids)

            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
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
        if self.is_instruct:
            return self._inference_instruct(samples)

        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        bos_embeds = self.word_embedding(
            self.tokenizer(self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            .input_ids[0].to(self.device)
        )
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            desc_ids = self._build_desc_ids(samples, i, descriptions)
            if self.dataset_name == 'fetaqa':
                input_ids = (desc_ids +
                           questions.input_ids[i][:fetaqa_question_len])
            else:
                input_ids = (desc_ids +
                           questions.input_ids[i])

            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred = [self._extract_response(p) for p in pred]

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

    @staticmethod
    def _extract_response(text):
        marker = '### Response:'
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
        text = text.strip()
        text = text.split('###')[0]
        text = text.split('\n')[0]
        return text.strip()

    @staticmethod
    def _strip_thinking(text):
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def _build_chat_prompt(self, desc_text, question_text, add_generation_prompt=True):
        user_content = desc_text + question_text
        for suffix in ('### Response:\n', '### Response:'):
            if user_content.endswith(suffix):
                user_content = user_content[:-len(suffix)]
                break

        messages = [{"role": "user", "content": user_content.strip()}]
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        needs_thinking_flag = (
            'qwen3' in self.args.llm_model_name.lower() or
            'gemma4' in self.args.llm_model_name.lower() or
            'gemma-4' in self.args.llm_model_name.lower()
        )
        if needs_thinking_flag and not self.enable_thinking:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _build_full_desc_text(self, samples, idx):
        desc_text = samples['desc'][idx]
        table_descs = samples.get('table_descs') or samples.get('table_segs')
        if table_descs is not None and table_descs[idx] is not None:
            tables_text = '\n'.join(table_descs[idx])
            marker = '### Input:\n'
            if marker in desc_text:
                prefix = desc_text[:desc_text.index(marker) + len(marker)]
                return prefix + tables_text
            return desc_text + tables_text
        return desc_text

    def _forward_instruct(self, samples):
        batch_size = len(samples['id'])
        eos_ids = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False).input_ids
        pad_id = self.tokenizer.pad_token_id

        all_input_ids, all_label_ids = [], []
        for i in range(batch_size):
            desc_text = self._build_full_desc_text(samples, i)
            prompt = self._build_chat_prompt(desc_text, samples['question'][i], add_generation_prompt=True)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_txt_len]

            label_ids = self.tokenizer(samples['label'][i], add_special_tokens=False).input_ids[:self.max_new_tokens]
            label_ids = label_ids + eos_ids

            all_input_ids.append(prompt_ids + label_ids)
            all_label_ids.append([IGNORE_INDEX] * len(prompt_ids) + label_ids)

        max_len = max(len(x) for x in all_input_ids)
        input_ids_padded, attention_mask, label_ids_padded = [], [], []
        for inp, lab in zip(all_input_ids, all_label_ids):
            pad_len = max_len - len(inp)
            input_ids_padded.append([pad_id] * pad_len + inp)
            attention_mask.append([0] * pad_len + [1] * len(inp))
            label_ids_padded.append([IGNORE_INDEX] * pad_len + lab)

        input_ids_t = torch.tensor(input_ids_padded).to(self.model.device)
        attn_mask_t = torch.tensor(attention_mask).to(self.model.device)
        label_ids_t = torch.tensor(label_ids_padded).to(self.model.device)

        with self.maybe_autocast():
            return causal_lm_loss_on_labels(
                self.model,
                input_ids=input_ids_t,
                attention_mask=attn_mask_t,
                labels=label_ids_t,
            )

    def _inference_instruct(self, samples):
        batch_size = len(samples['id'])
        prompts = []
        for i in range(batch_size):
            desc_text = self._build_full_desc_text(samples, i)
            prompts.append(self._build_chat_prompt(desc_text, samples['question'][i], add_generation_prompt=True))

        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            )

        new_tokens = outputs[:, input_ids.shape[1]:]
        pred = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        result = {
            'id': samples['id'],
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        print("\n" + "=" * 92)
        print(f"| {'COMPONENT':<22} | {'TOTAL PARAMS':>15} | {'TRAINABLE':>15} | {'% TRAINABLE':>11} |")
        print("|" + "-" * 90 + "|")
        print(f"| {'Baseline LLM':<22} | {total_params:>15,} | {trainable_params:>15,} | {pct:>10.2f}% |")
        print("=" * 92 + "\n")

        return trainable_params, total_params
