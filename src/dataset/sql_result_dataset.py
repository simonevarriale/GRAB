import os
import pandas as pd
from io import StringIO
from torch.utils.data import Dataset

import datasets as hf_datasets
from src.global_path import data_dir


def _find_dataset_path(name):
    p = os.path.join(data_dir, name, name)
    if os.path.isdir(p):
        return p
    raise FileNotFoundError(f"Dataset '{name}' not found under {data_dir}")


def _format_val(v):
    if isinstance(v, float) and not (v != v) and v == int(v):
        return str(int(v))
    return str(v)


def _linearize_table_df(table_name, df):
    cols = df.columns.tolist()
    segments = [f'[TAB] <table_name> : {table_name} col : {" | ".join(str(c) for c in cols)}']
    for row in df.itertuples(index=False):
        segments.append('[SEP] | ' + ' | '.join(_format_val(v) for v in row))
    return ' | '.join(segments)


def _linearize_table(table_name, table_json_str):
    df = pd.read_json(StringIO(table_json_str), orient='split')
    return _linearize_table_df(table_name, df)


_INSTRUCTION = (
    "You are a table question answering assistant. "
    "You are given one or more database tables and a natural language question. "
    "Your task is to generate the result table that answers the question. "
    "Output the result table using this exact format: "
    "col : header1 | header2 row 1 : val1 | val2 row 2 : val3 | val4. "
    "If the question cannot be answered using the given tables, return 'No Answer'. "
    "Do not provide any explanations or intermediate steps. "
    "Return only the result table."
)


class SQLResultDataset(Dataset):
    """Base class for result-table generation datasets (ATIS, GeoQuery, Spider-SQL).

    Each sample has a natural-language question over one or more DB tables and
    the expected output is the linearized result table produced by the SQL query.
    """

    def __init__(self, type, prompt_type='qwen', multi_table=True,
                 max_rows_per_table=None, dataset_name=None, **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type

        if isinstance(multi_table, str):
            self.multi_table = False if multi_table == 'False' else multi_table
        else:
            self.multi_table = multi_table

        path = _find_dataset_path(dataset_name)
        raw = hf_datasets.load_from_disk(path)

        if type not in raw:
            print(f"No '{type}' split for {dataset_name}, falling back to 'validation'")
            raw = raw['validation']
        else:
            raw = raw[type]

        print(f'Loaded {len(raw)} samples from disk for {dataset_name} split={type}')

        if self.multi_table == 'only':
            print('Filtering to multi-table examples only...')
            raw = raw.filter(lambda x: len(x['table_names']) > 1)
            self.multi_table = True
        elif not self.multi_table:
            print('Filtering to single-table examples only...')
            raw = raw.filter(lambda x: len(x['table_names']) == 1)

        if max_rows_per_table is not None:
            before = len(raw)

            def _within_limit(x):
                for t_json in x['tables']:
                    df = pd.read_json(StringIO(t_json), orient='split')
                    if len(df) > max_rows_per_table:
                        return False
                return True

            raw = raw.filter(_within_limit)
            print(f'Filtered {before - len(raw)} samples with a table exceeding '
                  f'{max_rows_per_table} rows ({len(raw)} remaining)')

        self.data = raw
        self.init_prompt = 'Please follow the instruction below.'

    def _make_prefix(self):
        if self.prompt_type == 'tablellama':
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_INSTRUCTION}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type in ('mistral', 'llama2', 'qwen'):
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_INSTRUCTION}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'small':
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_INSTRUCTION}\n\n"
                "Rules:\n"
                "- Output ONLY the result table in the specified format.\n"
                "- Do NOT explain.\n"
                "- Do NOT repeat the question.\n"
                "- Do NOT add any extra words.\n\n"
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
        segs = [_linearize_table_df(name, df) for name, df in zip(table_names, dfs)]
        table = dfs if self.multi_table else dfs[0]

        sample = {
            'id':          index,
            'question':    question,
            'label':       item['target'],
            'desc':        prefix,
            'table':       table,
            'table_segs':  segs,
            'table_names': table_names,
        }
        return sample
