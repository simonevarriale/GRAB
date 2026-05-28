import io
import pyarrow as pa
import pandas as pd
from torch.utils.data import Dataset

from src.global_path import data_dir

_ARROW_PATH = f'{data_dir}/hctqa/hctqa/stress_test/stress_test.arrow'

_INSTRUCTION = """
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


class HCTQAStressTestDataset(Dataset):
    def __init__(self, type='test', prompt_type='qwen', **kwargs):
        super().__init__()
        self.prompt_type = prompt_type
        self.init_prompt = 'Please follow the instruction below.'

        with pa.memory_map(_ARROW_PATH, 'r') as f:
            reader = pa.ipc.open_file(f)
            table = reader.read_all()

        self.data = table.to_pylist()

        self._table_cache = {}
        self.input_seg = []
        for row in self.data:
            table_id = row['table_id']
            if table_id not in self._table_cache:
                df = pd.read_csv(io.StringIO(row['table_as_csv']))
                cols = df.columns.tolist()
                segs = ['[TAB] | ' + ' | '.join(str(c) for c in cols)]
                for _, r in df.iterrows():
                    segs.append('[SEP] | ' + ' | '.join(str(v) for v in r))
                self._table_cache[table_id] = (' | '.join(segs), df)
            self.input_seg.append(self._table_cache[table_id][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        table_id = row['table_id']
        _, table_df = self._table_cache[table_id]
        label = row['answer']

        if self.prompt_type in ('qwen', 'tablellama'):
            question = f'### Question:\n{row["question"]}\n\n### Response:'
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "Please provide the answer using the shortest possible keywords, no additional context and explanation required.\n\n"
                f"### Instruction:\n{_INSTRUCTION}\n\n"
                "### Input:\n"
            )
        elif self.prompt_type in ('mistral', 'llama2', 'small'):
            question = f'### Question:\n{row["question"]}\n\n### Response:\n'
            desc = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{_INSTRUCTION}\n\n"
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
            'question_type': row['question_template_for_synthetic_only'],
        }
