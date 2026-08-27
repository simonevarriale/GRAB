import io
import os
import pandas as pd
from torch.utils.data import Dataset

import datasets

from src.global_path import data_dir
from src.dataset.table_utils import parse_html_table, num_header_rows, merge_header_rows, extract_html_caption


splits = ["train", "test", "validation"]


def _parse_html_table(html):
    return parse_html_table(html)


def _num_header_rows(html):
    return num_header_rows(html)


def _merge_header_rows(grid, n_header):
    return merge_header_rows(grid, n_header)


class HCTQADataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.datas = datasets.load_from_disk(f'{data_dir}/hctqa/hctqa')
        self.instruction = """
            You are a table question answering assistant.
            Your task is to answer the question based on the information in the table.
            The table structure may be complex and not a standard relational table so try to understand the structure of the table when answering the question.
            If the question cannot be answered using information from the table, return 'No Answer'.
            Do not provide any explanations, equations, code, or text explaining intermediate steps in figuring out the answer.
            

            If there are multiple columns that contain the answer, return the answer in this format:
            {val1 | val2} || {val3 | val4}

            Each {…} is one row result, || separates multiple rows, and | separates values within a row.
            Row order does not matter.

            Return only the final answer itself.
            """
        
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.datas[self.type])

    def _get_linear_table(self):
        self._table_cache = {}
        self.input_seg = []
        for data in self.datas[self.type]:
            table_id = data["table_id"]
            if table_id not in self._table_cache:
                html = data["table_as_html"]
                grid = _parse_html_table(html)
                if grid:
                    grid = _merge_header_rows(grid, _num_header_rows(html))
                    rows = ['[TAB] | ' + ' | '.join(grid[0])]
                    for row in grid[1:]:
                        rows.append('[SEP] | ' + ' | '.join(row))
                    linear = ' | '.join(rows)
                    caption = extract_html_caption(html)
                    if caption:
                        linear = f'[TLE] {caption} | ' + linear
                    self._table_cache[table_id] = linear
                else:
                    self._table_cache[table_id] = ''
            self.input_seg.append(self._table_cache[table_id])


    def __getitem__(self, index):
        data = self.datas[self.type][index]
        html = data["table_as_html"]
        grid = _parse_html_table(html)
        if grid:
            grid = _merge_header_rows(grid, _num_header_rows(html))
            table_df = pd.DataFrame(grid[1:], columns=[str(c) for c in grid[0]])
        else:
            table_df = pd.DataFrame()

        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = data['answer'] if isinstance(data['answer'], str) else ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )

        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = data['answer'] if isinstance(data['answer'], str) else ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )

        elif self.prompt_type == 'qwen':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = data['answer'] if isinstance(data['answer'], str) else ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )

        elif self.prompt_type == 'small':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = data['answer'] if isinstance(data['answer'], str) else ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
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

