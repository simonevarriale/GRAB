"""Check max question length in tokens across datasets.

Usage:
    python src/check_question_tokens.py
    python src/check_question_tokens.py --tokenizer ./models/Qwen/Qwen3-Embedding-0.6B
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.dataset import load_dataset

DATASETS = [
    "structProbe",
    "hitab",
    "wtq",
    "wikisql",
    "hctqa",
    "tabmwp",
    "multihiertt",
    "scitat",
    "mmqa",
    "tqa_bench",
    "atis",
    "geoquery",
    "spider_sql",
]

SPLITS = ["train", "validation", "test"]


def measure_dataset(name, tokenizer, prompt_type):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    global_max = 0
    global_max_text = ""

    for split in SPLITS:
        try:
            ds = load_dataset[name](split, prompt_type=prompt_type)
        except Exception:
            try:
                ds = load_dataset[name](split)
            except Exception as e:
                print(f"  {split:12s}: SKIP ({e})")
                continue

        if len(ds) == 0:
            print(f"  {split:12s}: empty")
            continue

        max_len = 0
        total_len = 0
        max_text = ""

        for i in range(len(ds)):
            sample = ds[i]
            question = sample["question"]
            n_tokens = len(tokenizer.encode(question, add_special_tokens=False))
            total_len += n_tokens
            if n_tokens > max_len:
                max_len = n_tokens
                max_text = question[:150]

        avg_len = total_len / len(ds)
        print(f"  {split:12s}: n={len(ds):6d}  max={max_len:5d}  avg={avg_len:7.1f}")

        if max_len > global_max:
            global_max = max_len
            global_max_text = max_text

    print(f"  {'GLOBAL MAX':12s}: {global_max:5d} tokens")
    print(f"  {'Sample':12s}: {global_max_text}")
    return name, global_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--prompt_type", type=str, default="qwen")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Prompt type: {args.prompt_type}")

    results = []
    for name in DATASETS:
        try:
            _, max_len = measure_dataset(name, tokenizer, args.prompt_type)
            results.append((name, max_len))
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Dataset: {name} — FAILED: {e}")
            results.append((name, -1))

    print(f"\n\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20s} {'Max Question Tokens':>20s}")
    print(f"{'-'*40}")
    for name, max_len in sorted(results, key=lambda x: -x[1]):
        val = str(max_len) if max_len >= 0 else "FAILED"
        print(f"{name:<20s} {val:>20s}")


if __name__ == "__main__":
    main()
