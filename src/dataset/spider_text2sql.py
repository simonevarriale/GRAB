import json
import random
import pandas as pd
from io import StringIO
from torch.utils.data import Dataset

import datasets as hf_datasets
from src.dataset.sql_result_dataset import _find_dataset_path, _linearize_table_df

_SPIDER_SEED = 42
_SPIDER_VAL_FRAC = 0.1

_SQL_INSTRUCTION = (
    "You are a text-to-SQL assistant. "
    "You are given one or more database tables and a natural language question. "
    "Your task is to generate the SQL query that answers the question. "
    "Return only the SQL query with no explanation or additional text."
)


class SpiderDataset(Dataset):
    """Spider dataset for text-to-SQL evaluation.

    Split strategy (Spider has no public test set):
      - train / validation : deterministic 90/10 carve of the HF train split
      - test               : the official HF validation (dev) split
    """

    def __init__(self, type, prompt_type='qwen', max_rows_per_table=None,
                 max_rows=None, **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        path = _find_dataset_path('spider')
        raw = hf_datasets.load_from_disk(path)

        if type == 'test':
            raw = raw['validation']
            self._id_map = list(range(len(raw)))
            self.precomputed_split = 'validation'
            print('Spider: test split = HF validation (held-out, not used for model selection)')
        elif type in ('train', 'validation'):
            full_train = raw['train']
            n = len(full_train)
            rng = random.Random(_SPIDER_SEED)
            idx = list(range(n))
            rng.shuffle(idx)
            n_val = max(1, int(n * _SPIDER_VAL_FRAC))
            subset = sorted(idx[:n_val] if type == 'validation' else idx[n_val:])
            self._id_map = subset
            raw = full_train.select(subset)
            self.precomputed_split = 'train'
            print(f'Spider re-split ({type}): {len(raw)} samples '
                  f'(90/10 carve of HF train, seed={_SPIDER_SEED})')
        else:
            raw = raw[type]
            self._id_map = list(range(len(raw)))
            self.precomputed_split = type

        print(f'Loaded {len(raw)} samples from spider split={type}')

        self.data = raw
        self.max_rows = max_rows_per_table if max_rows_per_table is not None else max_rows
        self.init_prompt = 'Please follow the instruction below.'

    def _make_prefix(self):
        if self.prompt_type == 'small':
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_SQL_INSTRUCTION}\n\n"
                "Rules:\n"
                "- Output ONLY the SQL query.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n\n"
                "### Input:\n"
            )
        elif self.prompt_type in ('tablellama', 'mistral', 'llama2', 'qwen'):
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_SQL_INSTRUCTION}\n\n"
                "### Input:\n"
            )
        else:
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if self.prompt_type in ('tablellama', 'qwen'):
            question = f'### Question:\n{item["question"]}\n\n### Response:'
        elif self.prompt_type in ('mistral', 'llama2', 'small'):
            question = f'### Question:\n{item["question"]}\n\n### Response:\n'
        else:
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

        prefix = self._make_prefix()
        table_names = item['table_names']
        tables_json = item['tables']

        dfs = [pd.read_json(StringIO(t_json), orient='split') for t_json in tables_json]
        if self.max_rows is not None:
            dfs = [df.iloc[:self.max_rows] for df in dfs]
        segs = [_linearize_table_df(name, df) for name, df in zip(table_names, dfs)]

        eval_meta = json.dumps({
            'gold_sql':    item['query'],
            'answer':      item['answer'],
            'tables':      item['tables'],
            'table_names': item['table_names'],
        })

        return {
            'id':         self._id_map[index],
            'question':   question,
            'label':      item['query'],
            'eval_meta':  eval_meta,
            'desc':       prefix,
            'table':      dfs,
            'table_segs': segs,
        }
