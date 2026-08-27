import ast
import pandas as pd
import datasets
from torch.utils.data import Dataset

from src.global_path import data_dir


class HiTabDataset(Dataset):
    def __init__(self, type, prompt_type='tablellama', **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        self.datas = datasets.load_from_disk(f'{data_dir}/hitab/hitab')
        self.instruction = 'This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical.'
        self._get_linear_table()

        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        return len(self.datas[self.type])

    def _fill_table_cell(self, table_text, merged_regions):
        for merged_region in merged_regions:
            cell_value = table_text[merged_region['first_row']][merged_region['first_column']]
            for i in range(merged_region['first_row'], merged_region['last_row'] + 1):
                for j in range(merged_region['first_column'],
                               min(len(table_text[0]), merged_region['last_column'] + 1)):
                    table_text[i][j] = cell_value
        return table_text

    def _get_linear_table(self):
        self.input_seg = []
        for data in self.datas[self.type]:
            table_content = ast.literal_eval(data['table_content'])
            table_array = self._fill_table_cell(table_content['texts'], table_content['merged_regions'])

            rows = []
            rows.append('[TAB] | ' + ' | '.join(table_array[0]))
            for row in table_array[1:]:
                rows.append('[SEP] | ' + ' | '.join(row))
            linear_table = ' | '.join(rows)
            linear_table = '[TLE] The table caption is ' + table_content['title'] + '. ' + linear_table
            self.input_seg.append(linear_table)

    def _get_table_df(self, data):
        table_content = ast.literal_eval(data['table_content'])
        texts = table_content['texts']
        merged_regions = table_content['merged_regions']

        for region in merged_regions:
            first_row, last_row = region['first_row'], region['last_row']
            first_col, last_col = region['first_column'], region['last_column']
            fill_value = texts[first_row][first_col]
            for i in range(first_row, last_row + 1):
                col_limit = min(len(texts[0]), last_col + 1)
                for j in range(first_col, col_limit):
                    texts[i][j] = fill_value

        if len(texts) > 0:
            return pd.DataFrame(texts[1:], columns=texts[0])
        else:
            return pd.DataFrame()

    def __getitem__(self, index):
        data = self.datas[self.type][index]
        data['answer'] = ast.literal_eval(data['answer'])
        data['answer'] = [str(i) for i in data['answer']]

        table_df = self._get_table_df(data)

        if self.prompt_type == 'tablellama':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'mistral' or self.prompt_type == 'llama2':
            question = f'### Question:\n{data["question"]}\n\n### Response:\n'
            label = ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'qwen':
            question = f'### Question:\n{data["question"]}\n\n### Response:'
            label = ', '.join(data['answer'])
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
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
            label = ', '.join(data['answer'])
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
        elif self.prompt_type == 'gemma_pt':
            question = f'Question: {data["question"]}\nAnswer:'
            label = ', '.join(data['answer'])
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
