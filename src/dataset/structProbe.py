import pandas as pd
import datasets
from torch.utils.data import Dataset

from src.global_path import data_dir


class StructProbeDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.datas = datasets.load_from_disk(f'{data_dir}/structProbe/structProbe')
        self.instruction = 'This is a table QA task. The goal of this task is to answer the question given the table.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        return len(self.datas[self.type])

    def _get_linear_table(self):
        self.input_seg = []
        for data in self.datas[self.type]:
            table_array = []
            table_array.append(data["table"]["header"])
            table_array.extend(data["table"]["rows"])
            linear_table = ''
            linear_table += 'col : ' + ' | '.join(table_array[0]) + ' [SEP]'
            for idx, row in enumerate(table_array[1:]):
                linear_table += ' row ' + row[0] + ' : ' + ' | '.join(row[1:]) + ' [SEP]'
            self.input_seg.append(linear_table)

    def __getitem__(self, index):
        data = self.datas[self.type][index]
        header = data["table"]["header"]
        rows = data["table"]["rows"]
        table_df = pd.DataFrame(rows, columns=header)

        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answers'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = ', '.join(data['answers'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'qwen':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answers'])
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
            label = ', '.join(data['answers'])
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
        elif self.prompt_type == 'gemma_pt':
            question = f'Question: {data["question"]}\nAnswer:'
            label = ', '.join(data['answers'])
            desc = (
                f"{self.instruction}\n\n"
                "Rules:\n"
                "- Output ONLY the answer.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n"
                "- Use the shortest possible answer.\n"
                "- If the answer contains multiple values, list them separated by \", \".\n\n"
                "Table:\n"
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
