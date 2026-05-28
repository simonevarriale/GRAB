import os
import json
import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir


class TabFactDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        split_file = 'val_examples.json' if type == 'validation' else f'{type}_examples.json'
        json_file = os.path.join(data_dir, 'tabfact', 'tabfact', split_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        table_dfs = {}
        linear_tables = {}

        for table_id in raw_data:
            df = self._read_table(table_id)
            table_dfs[table_id] = df
            linear_tables[table_id] = self._linearize_table(df)

        self.datas = []
        self.input_seg = []
        self.table_dfs = []

        for table_id, (statements, labels, caption) in raw_data.items():
            linear = linear_tables[table_id]
            linear_with_name = (f'[TLE] {caption} | ' if caption else '') + linear
            df = table_dfs[table_id]
            for statement, label in zip(statements, labels):
                self.datas.append({
                    'table_id': table_id,
                    'statement': statement,
                    'label': label,
                    'caption': caption,
                })
                self.input_seg.append(linear_with_name)
                self.table_dfs.append(df)

        self.instruction = (
            'This is a table-based fact verification task. '
            'The goal is to determine whether the given statement is supported (entailed) or refuted by the table. '
            'Answer with "1" for entailed or "0" for refuted.'
        )
        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        return len(self.datas)

    def _read_table(self, table_id):
        table_path = os.path.join(data_dir, 'tabfact', 'tabfact', 'all_csv', table_id)
        try:
            return pd.read_csv(table_path, sep='#', encoding='utf-8', engine='python')
        except Exception:
            try:
                return pd.read_csv(table_path, sep='#', encoding='latin-1', engine='python')
            except Exception:
                return pd.read_csv(table_path, sep=None, engine='python')

    @staticmethod
    def _linearize_table(table_df):
        header = table_df.columns.tolist()
        rows = ['[TAB] col : ' + ' | '.join(str(x) for x in header)]
        for _, row in table_df.iterrows():
            rows.append('[SEP] | ' + ' | '.join(str(x) for x in row))
        return ' | '.join(rows)

    def __getitem__(self, index):
        data = self.datas[index]
        table_df = self.table_dfs[index]
        statement = data['statement']
        label_value = data['label']
        caption = data['caption']
        table_id = data['table_id']

        if self.prompt_type == 'tablellama':
            question = f'### Question:\nIs the following statement entailed or refuted by the table?\nStatement: {statement}\n\n### Response:'
            label = str(label_value)
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\nIs the following statement entailed or refuted by the table?\nStatement: {statement}\n\n### Response:\n'
            label = str(label_value)
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer as either '1' (entailed) or '0' (refuted), no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'qwen':
            question = f'### Question:\nIs the following statement entailed or refuted by the table?\nStatement: {statement}\n\n### Response:'
            label = str(label_value)
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer as either '1' (entailed) or '0' (refuted), no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "Rules:\n"
                "- Output ONLY '1' for entailed or '0' for refuted.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n"
                "- Use only the number.\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'small':
            question = f'### Question:\nIs the following statement entailed or refuted by the table?\nStatement: {statement}\n\n### Response:\n'
            label = str(label_value)
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "Rules:\n"
                "- Output ONLY '1' for entailed or '0' for refuted.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n"
                "- Use only the number.\n\n"
                "### Input:\n"
            )
        else:
            raise ValueError(f'prompt_type {self.prompt_type} is not supported')

        return {
            'id': index,
            'table_id': table_id,
            'statement': statement,
            'caption': caption,
            'question': question,
            'label': label,
            'desc': desc,
            'table': table_df,
            'table_segs': [self.input_seg[index] + "\n\n"],
        }
