import json
import re
import os
import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir

_DATA_DIR = os.path.join(data_dir, 'scitat', 'scitat')

_SPLIT_FILES = {
    'train':      'scitat_train.json',
    'validation': 'scitat_dev.json',
    'test':       'scitat_test.json',
}


def _strip_latex(text: str) -> str:
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()


def _linearize_table(table: list) -> str:
    if not table:
        return ''
    header = [_strip_latex(str(c)) for c in table[0]]
    parts = ['[TAB] col : ' + ' | '.join(header)]
    for row in table[1:]:
        parts.append('[SEP] | ' + ' | '.join(_strip_latex(str(v)) for v in row))
    return ' | '.join(parts)


def _table_to_df(table: list) -> pd.DataFrame:
    if not table or len(table) < 2:
        return pd.DataFrame()
    header = [_strip_latex(str(c)) for c in table[0]]
    rows = [[_strip_latex(str(v)) for v in row] for row in table[1:]]
    return pd.DataFrame(rows, columns=header)


class SciTabDataset(Dataset):
    def __init__(self, type: str, prompt_type: str = 'qwen',
                 multi_table=True, **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type
        self.multi_table = multi_table if isinstance(multi_table, bool) \
            else (multi_table != 'False')

        with open(os.path.join(_DATA_DIR, _SPLIT_FILES[type]), encoding='utf-8') as f:
            self.datas = json.load(f)

        if not self.multi_table:
            self.datas = [s for s in self.datas if len(s['tables']) == 1]

        self.instruction = (
            'You are a scientific table question answering assistant. '
            'You are given one or more tables and a paragraph from a scientific paper and a question. '
            'Your task is to answer the question based on the information in the tables and the paragraph. '
            'The answer may require arithmetic operations on values from the tables or paragraph. '
            'If the question cannot be answered using the given tables and paragraph, return "No Answer". '
            'If there are multiple questions, answer them one by one and separate the answers with ". ". '
            'Do not provide any explanations or intermediate steps. '
            'Return only the final answer.'
        )
        self.init_prompt = 'Please follow the instruction below.'

        self._build_linearizations()

    def __len__(self):
        return len(self.datas)

    def _build_linearizations(self):
        self.input_segs = []
        self.table_segs_list = []
        for sample in self.datas:
            segs = [_linearize_table(t['table']) for t in sample['tables']]
            self.table_segs_list.append(segs)
            self.input_segs.append('\n'.join(segs))

    def _build_desc(self, index: int, context_block: str = '') -> tuple:
        if self.prompt_type == 'tablellama':
            prefix = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type in ('mistral', 'llama2', 'qwen'):
            prefix = (
                'Below is an instruction that describes a task, paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type == 'small':
            prefix = (
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
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

        segs = self.table_segs_list[index]
        if self.multi_table:
            return prefix + context_block, segs
        else:
            return prefix + context_block + segs[0] + '\n\n', None

    def __getitem__(self, index):
        sample = self.datas[index]

        answer = sample.get('answer', '')
        answer_list = answer if isinstance(answer, list) else [str(answer)]
        label = json.dumps([str(a) for a in answer_list])

        paragraph_text = _strip_latex(sample.get('paragraph', {}).get('text', ''))
        context_block = f'### Context:\n{paragraph_text}\n\n' if paragraph_text else ''

        if self.prompt_type in ('tablellama', 'qwen'):
            question = f'### Question:\n{sample["question"]}\n\n### Response:'
        else:
            question = f'### Question:\n{sample["question"]}\n\n### Response:\n'

        desc, table_segs = self._build_desc(index, context_block)

        dfs = [_table_to_df(t['table']) for t in sample['tables']]
        table = dfs if self.multi_table else dfs[0]

        item = {
            'id':       index,
            'question': question,
            'label':    label,
            'desc':     desc,
            'table':    table,
        }
        if table_segs is not None:
            item['table_segs'] = table_segs
        return item
