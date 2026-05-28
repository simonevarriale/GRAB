"""
Count Qwen3-Embedding-0.6B tokens for each row and column header across all datasets.

Uses load_dataset directly so that skip-list indices are consistent with training.
Row format mirrors TableTokenizer._build_row_strings: "col1: val1 col2: val2 ..."
Column format: each column name tokenized independently.

Run from the repo root:
    python src/count_row_col_tokens.py [--splits train validation test] [--datasets all|wtq|...]
    python src/count_row_col_tokens.py --skip_list skip_list.json --output results_row_col.json
"""

import argparse
import sys
import os
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.global_path import data_dir, model_dir
from src.dataset import load_dataset as ds_registry
from src.dataset.precomputed_wrapper import load_skip_set

MODEL_NAME = f"{model_dir}/Qwen/Qwen3-Embedding-0.6B"

_DEFAULT_DATASETS = [
    "structProbe", "hitab", "wtq", "wikisql", "hctqa", "tabmwp",
    "multihiertt", "mmqa", "scitat", "tqa_bench",
    "atis", "geoquery", "spider_sql",
]


def _tok(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _df_counts(tokenizer, df):
    columns = list(df.columns)
    col_counts = [_tok(tokenizer, str(c)) for c in columns]
    row_counts = []
    for r in range(len(df)):
        parts = [f"{col}: {df.iloc[r, c]}" for c, col in enumerate(columns)]
        row_counts.append(_tok(tokenizer, " ".join(parts)))
    return row_counts, col_counts


def stats(flat):
    if not flat:
        return {"count": 0, "min": 0, "p25": 0, "median": 0, "mean": 0.0,
                "p75": 0, "p90": 0, "p95": 0, "max": 0}
    a = np.array(flat, dtype=float)
    return {
        "count":  int(len(a)),
        "min":    int(a.min()),
        "p25":    int(np.percentile(a, 25)),
        "median": int(np.median(a)),
        "mean":   round(float(a.mean()), 1),
        "p75":    int(np.percentile(a, 75)),
        "p90":    int(np.percentile(a, 90)),
        "p95":    int(np.percentile(a, 95)),
        "max":    int(a.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",    nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--datasets",  nargs="+", default=["all"])
    parser.add_argument("--skip_list", default=None, help="Path to skip_list.json")
    parser.add_argument("--output",    default=None, help="Optional JSON output path")
    args = parser.parse_args()

    target_datasets = _DEFAULT_DATASETS if args.datasets == ["all"] else args.datasets

    print(f"Loading tokenizer: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Tokenizer loaded.\n")

    results = {}
    row_maxes, col_maxes = {}, {}

    for ds_name in target_datasets:
        if ds_name not in ds_registry:
            print(f"[WARN] Unknown dataset '{ds_name}', skipping.")
            continue

        results[ds_name] = {}
        ds_row_max = ds_col_max = 0

        for split in args.splits:
            print(f"{'='*60}")
            print(f"Dataset: {ds_name}  |  Split: {split}")

            try:
                ds = ds_registry[ds_name](split, prompt_type='qwen')
            except Exception as e:
                print(f"  [ERROR] loading dataset: {e}")
                continue

            skip_set = set()
            if args.skip_list:
                skip_set = load_skip_set(args.skip_list, ds_name, split)
                if skip_set:
                    print(f"  Skipping {len(skip_set)} samples from skip list.")

            row_counts, col_counts = [], []
            for idx in range(len(ds)):
                if idx in skip_set:
                    continue
                try:
                    sample = ds[idx]
                except Exception as e:
                    print(f"  [WARN] Sample {idx} failed: {e}")
                    continue

                table = sample.get('table')
                if table is None:
                    continue

                tables = table if isinstance(table, list) else [table]
                for df in tables:
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    r, c = _df_counts(tokenizer, df.astype(str))
                    row_counts.extend(r)
                    col_counts.extend(c)

            sr = stats(row_counts)
            sc = stats(col_counts)

            print(f"  --- Rows (format: 'col: val col: val ...') ---")
            print(f"  Count:                  {sr['count']}")
            print(f"  min / p25 / med / p75:  {sr['min']} / {sr['p25']} / {sr['median']} / {sr['p75']}")
            print(f"  p90 / p95 / max:        {sr['p90']} / {sr['p95']} / {sr['max']}")
            print(f"  mean:                   {sr['mean']}")
            print(f"  --- Column headers ---")
            print(f"  Count:                  {sc['count']}")
            print(f"  min / p25 / med / p75:  {sc['min']} / {sc['p25']} / {sc['median']} / {sc['p75']}")
            print(f"  p90 / p95 / max:        {sc['p90']} / {sc['p95']} / {sc['max']}")
            print(f"  mean:                   {sc['mean']}")

            results[ds_name][split] = {"rows": sr, "cols": sc}
            ds_row_max = max(ds_row_max, sr.get("max", 0))
            ds_col_max = max(ds_col_max, sc.get("max", 0))

        row_maxes[ds_name] = ds_row_max
        col_maxes[ds_name] = ds_col_max

    print(f"\n{'='*60}")
    print("MAX TOKENS PER DATASET (across all splits):")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20}  {'row_max_len':>12}  {'max_header_len':>14}")
    for ds_name in row_maxes:
        print(f"  {ds_name:<20}  {row_maxes[ds_name]:>12}  {col_maxes[ds_name]:>14}")
    overall_row = max(row_maxes.values()) if row_maxes else 0
    overall_col = max(col_maxes.values()) if col_maxes else 0
    print(f"\n  {'Overall max':<20}  {overall_row:>12}  {overall_col:>14}")

    if args.output:
        out = {
            "per_dataset":         results,
            "max_row_per_dataset": row_maxes,
            "max_col_per_dataset": col_maxes,
            "overall_row_max":     overall_row,
            "overall_col_max":     overall_col,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
