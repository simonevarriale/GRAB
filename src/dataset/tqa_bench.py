import os
import random
import sqlite3

import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir

_DATA_DIR = os.path.join(data_dir, 'tqa_bench', 'tqa_bench')
_OPTION_LETTERS = ['A', 'B', 'C', 'D']
_ALL_DOMAINS = [
    'airline', 'cookbook', 'food_facility_inspections', 'food_inspection',
    'global_biodiversity', 'movie', 'music_tracker', 'restaurant',
    'university', 'water_quality',
]


class TQABenchDataset(Dataset):
    """TQA-Bench: multi-table multiple-choice QA benchmark."""

    def __init__(self, type: str, prompt_type: str = 'qwen',
                 scale: str = '8k', domain=None,
                 seed: int = 42, train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 multi_table=True, **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type
        self.scale = scale
        self.multi_table = multi_table if isinstance(multi_table, bool) \
            else (multi_table != 'False')

        domains = _ALL_DOMAINS if domain is None else (
            [domain] if isinstance(domain, str) else list(domain)
        )

        conn = sqlite3.connect(os.path.join(_DATA_DIR, 'dataset.sqlite'))
        rows = []
        for dom in domains:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT scale, dbIdx, sampleIdx, questionIdx, qtype, '
                f'question, rightIdx, A, B, C, D FROM "{dom}" WHERE scale=?',
                (scale,)
            )
            for r in cursor.fetchall():
                rows.append({
                    'domain': dom, 'scale': r[0], 'dbIdx': r[1],
                    'sampleIdx': r[2], 'questionIdx': r[3],
                    'qtype': r[4], 'question': r[5], 'rightIdx': r[6],
                    'A': r[7], 'B': r[8], 'C': r[9], 'D': r[10],
                })
        conn.close()

        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        split_map = {
            'train':      indices[:n_train],
            'validation': indices[n_train:n_train + n_val],
            'test':       indices[n_train + n_val:],
        }
        self.data = [rows[i] for i in split_map[type]]

        self._db_cache: dict = {}

        self.instruction = (
            'You are a multi-table question answering assistant. '
            'You are given one or more database tables and a multiple-choice question. '
            'Your task is to select the correct answer (A, B, C, or D) '
            'based on the information in the tables. '
            'Return only the letter of the correct option.'
        )
        self.init_prompt = 'Please follow the instruction below.'

        self._build_linearizations()

    def __len__(self):
        return len(self.data)

    def _load_tables(self, domain: str, db_idx: int):
        key = (self.scale, domain, db_idx)
        if key not in self._db_cache:
            db_path = os.path.join(_DATA_DIR, self.scale, domain, f'{db_idx}.sqlite')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            names = [r[0] for r in cursor.fetchall()]
            tables = [(n, pd.read_sql_query(f'SELECT * FROM "{n}"', conn)) for n in names]
            conn.close()
            self._db_cache[key] = tables
        return self._db_cache[key]

    @staticmethod
    def _linearize(table_name: str, df: pd.DataFrame) -> str:
        cols = [str(c) for c in df.columns]
        parts = [f'[TAB] {table_name} col : ' + ' | '.join(cols)]
        for row in df.itertuples(index=False, name=None):
            parts.append('[SEP] | ' + ' | '.join(str(v) for v in row))
        return ' | '.join(parts)

    def _build_linearizations(self):
        self.table_segs_list = []
        self.input_segs = []
        for sample in self.data:
            tables = self._load_tables(sample['domain'], sample['dbIdx'])
            segs = [self._linearize(name, df) for name, df in tables]
            self.table_segs_list.append(segs)
            self.input_segs.append('\n'.join(segs))

    def _build_desc(self, index: int):
        if self.prompt_type == 'tablellama':
            prefix = (
                'Below is an instruction that describes a task, paired with an input '
                'that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type in ('mistral', 'llama2', 'qwen'):
            prefix = (
                'Below is an instruction that describes a task, paired with an input '
                'that provides further context. '
                'Write a response that appropriately completes the request. '
                'Return only the letter of the correct option (A, B, C, or D), '
                'no additional context or explanation required.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                '### Input:\n'
            )
        elif self.prompt_type == 'small':
            prefix = (
                'Below is an instruction that describes a task, paired with an input '
                'that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                f'### Instruction:\n{self.instruction}\n\n'
                'Rules:\n'
                '- Output ONLY the letter (A, B, C, or D).\n'
                '- Do NOT explain.\n'
                '- Do NOT repeat the question.\n'
                '- Do NOT add any extra words.\n\n'
                '### Input:\n'
            )
        else:
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

        if self.multi_table:
            return prefix, self.table_segs_list[index]
        else:
            return prefix + self.input_segs[index] + '\n\n', None

    def __getitem__(self, index: int):
        sample = self.data[index]

        options = (f"A. {sample['A']}\nB. {sample['B']}\n"
                   f"C. {sample['C']}\nD. {sample['D']}")
        q_text = f"{sample['question']}\n\n{options}"

        if self.prompt_type in ('tablellama', 'qwen'):
            question = f'### Question:\n{q_text}\n\n### Response:'
        else:
            question = f'### Question:\n{q_text}\n\n### Response:\n'

        label = _OPTION_LETTERS[sample['rightIdx']]
        desc, table_segs = self._build_desc(index)

        tables = self._load_tables(sample['domain'], sample['dbIdx'])
        dfs = [df for _, df in tables]
        table = dfs if self.multi_table else dfs[0]

        item = {
            'id':            index,
            'question':      question,
            'label':         label,
            'desc':          desc,
            'table':         table,
            'question_type': sample['qtype'],
            'domain':        sample['domain'],
        }
        if table_segs is not None:
            item['table_segs'] = table_segs
        return item
