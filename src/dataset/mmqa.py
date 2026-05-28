import ast
import pandas as pd
import datasets
from torch.utils.data import Dataset

from src.global_path import data_dir


_INSTRUCTION = (
    "You are a table question answering assistant. "
    "You are given one or more database tables and a question. "
    "Your task is to answer the question based on the information in the tables. "
    "If the answer contains multiple values, list them separated by ' | '. "
    "If the question cannot be answered using the given tables, return 'No Answer'. "
    "Do not provide any explanations, equations, code, or intermediate steps. "
    "Return only the final answer."
)


def _parse_mmqa_value(value):
    s = str(value).strip()

    if s.startswith('{'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict) and 'data' in parsed:
                values = []
                for row in parsed['data']:
                    if isinstance(row, (list, tuple)):
                        values.extend(str(v) for v in row)
                    else:
                        values.append(str(row))
                return values
        except (ValueError, SyntaxError):
            pass
        return [s]

    if s.startswith('['):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        return [s]

    return [s]


class MMQADataset(Dataset):
    def __init__(self, type, prompt_type='qwen', seed=42,
                 train_ratio=0.70, val_ratio=0.15,
                 multi_table=True, original_split_only=None,
                 max_rows_per_table=None,
                 **kwargs):
        super().__init__()
        self.type = type
        self.prompt_type = prompt_type
        if isinstance(multi_table, bool):
            self.multi_table = multi_table
        else:
            self.multi_table = False if multi_table == 'False' else multi_table

        raw = datasets.load_from_disk(f'{data_dir}/mmqa/mmqa')[type]

        print(f'Loaded {len(raw)} samples from disk for split={type}')

        if self.multi_table == 'only':
            self.multi_table = True
        elif not self.multi_table:
            raw = raw.filter(lambda x: x['question_type_ext'] == 'single_table')

        if max_rows_per_table is not None:
            before = len(raw)
            raw = raw.filter(lambda x: all(len(t) <= max_rows_per_table for t in x['tables_rows']))
            print(f'Filtered {before - len(raw)} samples with a table exceeding {max_rows_per_table} rows '
                  f'({len(raw)} remaining)')

        self.data = raw
        self.instruction = _INSTRUCTION
        self.init_prompt = 'Please follow the instruction below.'

    def __len__(self):
        return len(self.data)

    def _build_desc(self, index, item):
        if self.prompt_type == 'tablellama':
            prefix = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type in ('mistral', 'llama2', 'qwen'):
            prefix = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{self.instruction}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type == 'small':
            prefix = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
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
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

        segs = item['tables_text']
        names = item.get('table_names', [])
        if names:
            segs = [
                (f'[TLE] {name} | ' if name else '') + seg
                for name, seg in zip(names, segs)
            ]
        return prefix, segs

    def __getitem__(self, index):
        item = self.data[index]

        if self.prompt_type in ('tablellama', 'qwen'):
            question = f'### Question:\n{item["question"]}\n\n### Response:'
        elif self.prompt_type in ('mistral', 'llama2', 'small'):
            question = f'### Question:\n{item["question"]}\n\n### Response:\n'
        else:
            raise ValueError(f'Unknown prompt_type: {self.prompt_type}')

        desc, table_segs = self._build_desc(index, item)

        cols_per_table = item['tables_columns']
        rows_per_table = item['tables_rows']
        dfs = [pd.DataFrame(rows, columns=cols)
               for cols, rows in zip(cols_per_table, rows_per_table)]
        table = dfs if self.multi_table else dfs[0]

        sample = {
            'id':            index,
            'question':      question,
            'label':         ' | '.join(_parse_mmqa_value(item['value'])),
            'desc':          desc,
            'table':         table,
            'question_type': item['question_type_ext'],
            'db_id':         item['db_id'],
            'query':         item['query'],
            'foreign_keys':  item.get('foreign_keys', []),
            'table_names':   item.get('table_names', []),
            'num_tables':    item.get('num_tables', len(dfs)),
            'source_variant': item.get('source_variant', ''),
        }
        sample['table_segs'] = table_segs
        return sample
