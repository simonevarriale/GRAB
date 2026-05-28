import json
import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir


class TabMWPDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        split_file = {'train': 'problems_train', 'dev': 'problems_dev', 'validation': 'problems_dev', 'test': 'problems_test'}
        path = f'{data_dir}/tabmwp/tabmwp/{split_file[type]}.json'
        raw = json.load(open(path))
        self.datas = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]

        self.instruction = (
            'This is a table math word problem task. '
            'The goal is to answer the question given the table.'
        )
        self.init_prompt = 'Please follow the instruction below.'
        self._get_linear_table()

    def __len__(self):
        return len(self.datas)

    def _parse_table(self, data):
        lines = data['table'].split('\n')
        header = [c.strip() for c in lines[0].split('|')]
        rows = [[c.strip() for c in line.split('|')] for line in lines[1:] if line.strip()]
        return header, rows

    def _get_linear_table(self):
        self.input_seg = []
        for data in self.datas:
            header, rows = self._parse_table(data)
            title = data.get('table_title', '')

            parts = []
            if title:
                parts.append(f'[TLE] {title}')
            parts.append('[TAB] col : ' + ' | '.join(header))
            for row in rows:
                parts.append('[SEP] | ' + ' | '.join(row))
            self.input_seg.append(' | '.join(parts))

    def __getitem__(self, index):
        data = self.datas[index]
        header, rows = self._parse_table(data)
        table_df = pd.DataFrame(rows, columns=header)

        question_text = data['question']
        if data.get('choices'):
            choices_str = ' '.join(f'({chr(65+i)}) {c}' for i, c in enumerate(data['choices']))
            question_text = f'{question_text} Choices: {choices_str}'

        label = str(data['answer'])

        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{question_text}\n\n### Response:'
            desc = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type in ('mistral', 'llama2'):
            question = f'### Question:\n{question_text}\n\n### Response:\n'
            desc = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request. '
                'Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type == 'qwen':
            question = f'### Question:\n{question_text}\n\n### Response:'
            desc = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                'Rules:\n'
                '- Output ONLY the answer.\n'
                '- Do NOT explain.\n'
                '- Do NOT repeat the question.\n'
                '- Do NOT add any extra words.\n'
                '- Use the shortest possible answer.\n\n'
                '### Input:\n'
            )
        elif self.prompt_type == 'small':
            question = f'### Question:\n{question_text}\n\n### Response:\n'
            desc = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                'Rules:\n'
                '- Output ONLY the answer.\n'
                '- Do NOT explain.\n'
                '- Do NOT repeat the question.\n'
                '- Do NOT add any extra words.\n'
                '- Use the shortest possible answer.\n\n'
                '### Input:\n'
            )
        else:
            raise ValueError(f'prompt_type {self.prompt_type} is not supported')

        return {
            'id': index,
            'question': question,
            'label': label,
            'desc': desc,
            'table': table_df,
            'table_segs': [self.input_seg[index] + "\n\n"],
        }
