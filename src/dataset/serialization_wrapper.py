import json

import pandas as pd
from torch.utils.data import Dataset

FORMATS = ("markdown", "csv", "json")


def serialize_df(df: pd.DataFrame, name: str, fmt: str) -> str:
    df = df.astype(str)
    cols = [str(c) for c in df.columns]
    title = f"Table: {name}\n" if name else ""

    if fmt == "markdown":
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        for row in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
        return title + "\n".join(lines)

    if fmt == "csv":
        return title + df.to_csv(index=False).rstrip("\n")

    if fmt == "json":
        records = [dict(zip(cols, (str(v) for v in row)))
                   for row in df.itertuples(index=False, name=None)]
        return title + json.dumps(records, ensure_ascii=False)

    raise ValueError(f"Unknown serialization format: {fmt} (choose from {FORMATS})")


class SerializationFormatDataset(Dataset):
    """Wraps a dataset and re-serializes `table_segs` in the given format."""

    def __init__(self, base_dataset, fmt: str):
        if fmt not in FORMATS:
            raise ValueError(f"Unknown serialization format: {fmt} (choose from {FORMATS})")
        self.base = base_dataset
        self.fmt = fmt
        self.init_prompt = getattr(base_dataset, "init_prompt", None)
        if hasattr(base_dataset, "precomputed_split"):
            self.precomputed_split = base_dataset.precomputed_split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        tables = sample.get("table")
        dfs = tables if isinstance(tables, list) else [tables]
        names = sample.get("table_names") or [""] * len(dfs)
        if len(names) != len(dfs):
            names = [""] * len(dfs)
        sample["table_segs"] = [
            serialize_df(df, str(name), self.fmt)
            for df, name in zip(dfs, names)
            if isinstance(df, pd.DataFrame)
        ]
        return sample
