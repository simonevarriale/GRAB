import json
import pandas as pd
import datasets
from torch.utils.data import Dataset

from src.global_path import data_dir


class WikiSQLDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.datas = datasets.load_from_disk(f'{data_dir}/wikisql/wikisql')
        self.instruction = 'This is a table QA task. The goal of this task is to answer the question given the table.'
        self._get_linear_table()
        with open(f'{data_dir}/wikisql/answers.json', 'r') as f:
            self.answers = json.load(f)[self.type]

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        return len(self.datas[self.type])

    def _get_linear_table(self):
        self.input_seg = []
        for data in self.datas[self.type]:
            table_array = []
            table_array.append(data["table"]["header"])
            table_array.extend(data["table"]["rows"])

            rows = []
            rows.append('[TAB] col : ' + ' | '.join(table_array[0]))
            for row in table_array[1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table = ' | '.join(rows)
            self.input_seg.append(linear_table)

    def __getitem__(self, index):
        data = self.datas[self.type][index]
        header = data["table"]["header"]
        rows = data["table"]["rows"]
        table_df = pd.DataFrame(rows, columns=header)

        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{data["question"].strip()}\n\n### Response:'
            label = ', '.join(self.answers[index])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\n{data["question"].strip()}\n\n### Response:\n'
            label = ', '.join(self.answers[index])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'qwen':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(self.answers[index])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "Rules:\n"
                "- Output ONLY the answer.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n"
                "- Use the shortest possible answer.\n"
                "- If the answer contains multiple values, list them separated by \", \".\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'small':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = ', '.join(self.answers[index])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                "This is a table QA task. The goal is to answer the question using ONLY the table.\n"
                "Rules:\n"
                "- Output ONLY the answer.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n"
                "- Use the shortest possible answer.\n\n"
                "### Input:\n"
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
