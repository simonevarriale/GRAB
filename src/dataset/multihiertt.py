import json
import os
import re
import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir
from src.dataset.table_utils import parse_html_table, num_header_rows, merge_header_rows

_DATA_DIR = os.path.join(data_dir, 'multihiertt', 'multihiertt')
_VAL_FRACTION = 0.1
_SPLIT_SEED = 42


def _linearize_table(grid, table_idx: int, html: str = '') -> str:
    if not grid:
        return ''
    if html:
        grid = merge_header_rows(grid, num_header_rows(html))
    rows = ['[TAB] | ' + ' | '.join(grid[0])]
    for row in grid[1:]:
        rows.append('[SEP] | ' + ' | '.join(row))
    return f'[TLE] Table {table_idx}: ' + ' | '.join(rows)


def _build_context(paragraphs) -> str:
    parts = []
    for p in paragraphs:
        cleaned = re.sub(r'##\s*Table\s*\d+\s*##', '', p).strip()
        if cleaned:
            parts.append(cleaned)
    return ' '.join(parts)


def _load_split(split: str):
    if split == 'test':
        with open(os.path.join(_DATA_DIR, 'dev.json'), 'r', encoding='utf-8') as f:
            return json.load(f)

    with open(os.path.join(_DATA_DIR, 'train.json'), 'r', encoding='utf-8') as f:
        all_train = json.load(f)

    import random
    rng = random.Random(_SPLIT_SEED)
    indices = list(range(len(all_train)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_train) * _VAL_FRACTION))
    val_indices = set(indices[:n_val])

    if split == 'validation':
        return [all_train[i] for i in range(len(all_train)) if i in val_indices]
    else:
        return [all_train[i] for i in range(len(all_train)) if i not in val_indices]


class MultiHierTTDataset(Dataset):
    def __init__(self, type: str, prompt_type: str = 'tablellama',
                 multi_table=True, max_rows_per_table=None, **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type
        self.multi_table = multi_table if isinstance(multi_table, bool) else (multi_table != 'False')

        self.datas = _load_split(type)

        if max_rows_per_table is not None:
            before = len(self.datas)
            self.datas = [
                s for s in self.datas
                if all(len(parse_html_table(t)) <= max_rows_per_table for t in s['tables'])
            ]
            print(f'Filtered {before - len(self.datas)} samples exceeding {max_rows_per_table} rows/table '
                  f'({len(self.datas)} remaining)')

        self.instruction = (
            'This is a multi-table hierarchical financial question answering task. '
            'You are given a financial document that contains multiple hierarchical tables (Table 0, Table 1, ...) '
            'and supporting text paragraphs. '
            'Each table may have multi-level headers expressed via merged cells (colspan/rowspan). '
            'The question may require reasoning across one or more tables, '
            'including arithmetic operations such as sum, average, or difference. '
            'Answer the question with the value or values that correspond to the answer. '
            "If the answer contains multiple values, list them separated by ' | '. "
            'Do not explain your reasoning.'
        )
        self.init_prompt = 'Please follow the instruction below.'

        self._build_linearizations()

    def __len__(self):
        return len(self.datas)

    def _build_linearizations(self):
        self.input_segs = []
        for sample in self.datas:
            htmls = sample['tables']
            grids = [parse_html_table(t) for t in htmls]
            table_parts = [_linearize_table(g, i, htmls[i]) for i, g in enumerate(grids) if g]
            context = _build_context(sample.get('paragraphs', []))
            if context:
                linear = f'[TLE] Context: {context} ' + ' '.join(table_parts)
            else:
                linear = ' '.join(table_parts)
            self.input_segs.append(linear)

    def _html_to_df(self, html: str) -> pd.DataFrame:
        grid = parse_html_table(html)
        if not grid:
            return pd.DataFrame()
        grid = merge_header_rows(grid, num_header_rows(html))
        return pd.DataFrame(grid[1:], columns=[str(c) for c in grid[0]])

    def _get_table_field(self, sample):
        if self.multi_table:
            return [self._html_to_df(t) for t in sample['tables']]

        table_evidence = sample.get('qa', {}).get('table_evidence', [])
        table_idx = 0
        if table_evidence:
            try:
                table_idx = int(table_evidence[0].split('-')[0])
            except (ValueError, IndexError):
                table_idx = 0
        table_idx = min(table_idx, len(sample['tables']) - 1)
        return self._html_to_df(sample['tables'][table_idx])

    def __getitem__(self, index):
        sample = self.datas[index]
        qa = sample.get('qa', {})
        question_text = qa.get('question', '')
        answer = qa.get('answer', '')
        label = str(answer) if answer != '' else ''

        table_field = self._get_table_field(sample)

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
                '- Use the shortest possible answer.\n'
                '- If the answer contains multiple values, list them separated by " | ".\n\n'
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
        elif self.prompt_type == 'gemma_pt':
            question = f'Question: {question_text}\nAnswer:'
            desc = (
                f'{self.instruction}\n\n'
                'Rules:\n'
                '- Output ONLY the answer.\n'
                '- Do NOT explain.\n'
                '- Do NOT repeat the question.\n'
                '- Do NOT add any extra words.\n'
                '- Use the shortest possible answer.\n'
                '- If the answer contains multiple values, list them separated by " | ".\n\n'
                'Tables:\n'
            )
        else:
            raise ValueError(f'prompt_type {self.prompt_type} is not supported')

        return {
            'id': index,
            'question': question,
            'label': label,
            'desc': desc,
            'table': table_field,
            'foreign_keys': None,
            'table_names': [str(i) for i in range(len(sample['tables']))],
            'table_segs': [self.input_segs[index] + "\n\n"],
        }
