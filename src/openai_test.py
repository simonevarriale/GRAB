"""
OpenAI inference script for table QA datasets.

Usage (online, parallel):
    python -m src.openai_test \
        --dataset wtq \
        --model gpt-4o-mini \
        --split test

Usage (Batch API, 50% cheaper, async):
    python -m src.openai_test \
        --dataset wtq \
        --model gpt-4o-mini \
        --split test \
        --batch

Resume a submitted batch (skip re-upload):
    python -m src.openai_test \
        --dataset wtq \
        --model gpt-4o-mini \
        --split test \
        --batch --batch_id batch_abc123
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(load_dataset.keys()))
    parser.add_argument("--openai_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model name")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--prompt_type", type=str, default="mistral")
    parser.add_argument("--skip_list", type=str, default="skip_list.json",
                        help="Path to skip_list.json (set to '' to disable)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: full dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API calls (online mode only)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in the output file (online mode only)")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API (50%% cheaper, async, up to 24h)")
    parser.add_argument("--batch_id", type=str, default=None,
                        help="Resume polling an already-submitted batch by its ID")
    return parser.parse_args()


_MODELS_USING_MAX_COMPLETION_TOKENS = ("o1", "o3", "o4", "gpt-5")
_MODELS_NO_SYSTEM_ROLE = ("o1", "o3", "o4", "gpt-5")
_REASONING_EFFORT_NONE = ("gpt-5.4",)


def _load_dataset(args):
    from src.dataset.precomputed_wrapper import SkipListDataset, load_skip_set
    dataset = load_dataset[args.dataset](
        args.split, prompt_type=args.prompt_type, multi_table=True,
    )
    if args.skip_list:
        dataset = SkipListDataset(dataset, load_skip_set(args.skip_list, args.dataset, args.split))
        print(f"After skip list: {len(dataset)} samples")
    return dataset


def build_prompt(item):
    desc = item["desc"].strip()
    table_segs = item.get("table_descs") or item.get("table_segs")
    if table_segs:
        table_text = "\n".join(table_segs) if isinstance(table_segs, list) else table_segs
        desc = desc + table_text.strip()
    system = desc
    user = item["question"].strip().removesuffix("### Response:").strip()
    return system, user


def _build_messages(model, system_prompt, user_prompt):
    no_system = any(model.startswith(p) for p in _MODELS_NO_SYSTEM_ROLE)
    if no_system:
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _reasoning_effort(model):
    if any(model.startswith(p) for p in _REASONING_EFFORT_NONE):
        return "none"
    return "minimal"


def _build_batch_body(model, messages, max_tokens):
    is_reasoning = any(model.startswith(p) for p in _MODELS_USING_MAX_COMPLETION_TOKENS)
    body = {"model": model, "messages": messages}
    if is_reasoning:
        body["max_completion_tokens"] = max_tokens
        body["reasoning_effort"] = _reasoning_effort(model)
    else:
        body["max_tokens"] = max_tokens
    return body


def call_openai(client, model, system_prompt, user_prompt, max_tokens, retries=3):
    messages = _build_messages(model, system_prompt, user_prompt)
    is_reasoning = any(model.startswith(p) for p in _MODELS_USING_MAX_COMPLETION_TOKENS)
    token_kwarg = (
        {"max_completion_tokens": max_tokens, "reasoning_effort": _reasoning_effort(model)}
        if is_reasoning else {"max_tokens": max_tokens}
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **token_kwarg,
            )
            choice = response.choices[0]
            content = choice.message.content
            return content.strip() if content else ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {retries} attempts: {e}")
                return ""


def run_batch_inference(args):
    if not args.openai_key:
        raise ValueError("No OpenAI API key provided. Use --openai_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=args.openai_key)
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)

    out_path = (
        f"{args.output_dir}/{args.dataset}/"
        f"openai_{args.model}_{args.dataset}_{args.split}.jsonl"
    )
    batch_id_path = (
        f"{args.output_dir}/{args.dataset}/"
        f"openai_{args.model}_{args.dataset}_{args.split}.batch_id"
    )

    batch_id = args.batch_id
    if batch_id is None and os.path.exists(batch_id_path):
        with open(batch_id_path) as f:
            batch_id = f.read().strip()
        print(f"Found saved batch ID: {batch_id}")

    if batch_id is None:
        dataset = _load_dataset(args)
        max_samples = args.max_samples if args.max_samples is not None else len(dataset)
        print(f"Loaded {args.dataset} {args.split}: {len(dataset)} samples (submitting {max_samples})")

        lines = []
        for i in range(max_samples):
            item = dataset[i]
            system_prompt, user_prompt = build_prompt(item)
            messages = _build_messages(args.model, system_prompt, user_prompt)
            body = _build_batch_body(args.model, messages, args.max_new_tokens)
            lines.append(json.dumps({
                "custom_id": str(item["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))

        jsonl_bytes = "\n".join(lines).encode()
        print(f"Uploading batch input file ({len(lines)} requests, {len(jsonl_bytes) / 1024:.1f} KB)...")
        file_obj = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        with open(batch_id_path, "w") as f:
            f.write(batch_id)
        print(f"Batch submitted: {batch_id}  (saved to {batch_id_path})")

    poll_interval = 60
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(
            f"[{time.strftime('%H:%M:%S')}] status={status}  "
            f"completed={counts.completed}  failed={counts.failed}  total={counts.total}"
        )
        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {status}")
        time.sleep(poll_interval)

    output_content = client.files.content(batch.output_file_id).text
    results_map = {}
    for line in output_content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        if obj.get("error") or obj["response"]["status_code"] != 200:
            pred = ""
        else:
            choices = obj["response"]["body"]["choices"]
            content = choices[0]["message"]["content"] if choices else ""
            pred = content.strip() if content else ""
        results_map[custom_id] = pred

    dataset = _load_dataset(args)
    max_samples = args.max_samples if args.max_samples is not None else len(dataset)

    results = []
    with open(out_path, "w") as f_out:
        for i in range(max_samples):
            item = dataset[i]
            row = {
                "id": item["id"],
                "pred": results_map.get(str(item["id"]), ""),
                "label": item["label"],
            }
            if "question_type" in item:
                row["question_type"] = item["question_type"]
            results.append(row)
            f_out.write(json.dumps(row) + "\n")

    if os.path.exists(batch_id_path):
        os.remove(batch_id_path)

    print(f"Results written to {out_path}")
    return results


def run_inference(args):
    if not args.openai_key:
        raise ValueError("No OpenAI API key provided. Use --openai_key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=args.openai_key)

    dataset = _load_dataset(args)
    max_samples = args.max_samples if args.max_samples is not None else len(dataset)
    print(f"Loaded {args.dataset} {args.split}: {len(dataset)} samples (evaluating {max_samples})")

    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    out_path = (
        f"{args.output_dir}/{args.dataset}/"
        f"openai_{args.model}_{args.dataset}_{args.split}.jsonl"
    )

    done_ids = set()
    existing_rows = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                done_ids.add(row["id"])
                existing_rows.append(row)
        print(f"Resuming: {len(done_ids)} samples already done")

    pending = [dataset[i] for i in range(max_samples) if i not in done_ids]

    if pending:
        sys_ex, usr_ex = build_prompt(pending[0])
        print("\n--- SAMPLE PROMPT ---")
        print(f"[SYSTEM]\n{sys_ex}\n")
        print(f"[USER]\n{usr_ex}")
        print("--- END PROMPT ---\n")

    def process(item):
        system_prompt, user_prompt = build_prompt(item)
        pred = call_openai(
            client, args.model,
            system_prompt, user_prompt,
            args.max_new_tokens,
        )
        row = {"id": item["id"], "pred": pred, "label": item["label"]}
        if "question_type" in item:
            row["question_type"] = item["question_type"]
        return row

    results = list(existing_rows)
    open_mode = "a" if args.resume else "w"
    with open(out_path, open_mode) as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, item): item for item in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
                row = future.result()
                results.append(row)
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()

    return results


def score_results(args, results):
    results.sort(key=lambda x: x["id"])
    eval_output = [{k: [v] for k, v in row.items()} for row in results]

    csv_path = (
        f"{args.output_dir}/{args.dataset}/"
        f"openai_{args.model}_{args.dataset}_{args.split}.csv"
    )
    score_txt = f"{args.output_dir}/{args.dataset}/score.txt"

    if args.dataset in eval_funcs:
        result = eval_funcs[args.dataset](eval_output, csv_path)
        with open(score_txt, "a", encoding="utf-8") as f:
            f.write(f"{csv_path}\n")
            if isinstance(result, dict) and "overall_acc" in result:
                print(f"{args.dataset} Test Acc: {result['overall_acc']:.4f}")
                f.write(f"Overall Acc: {result['overall_acc']:.4f}\n")
                for key, val in result.items():
                    if key != "overall_acc":
                        f.write(f"{key}: acc={val['acc']:.4f}, count={val['count']}\n")
            elif isinstance(result, dict) and "table_em" in result:
                print(
                    f"{args.dataset} Table EM: {result['table_em']:.4f}, "
                    f"Row F1: {result['row_em_f1']:.4f}, "
                    f"Col F1: {result['col_em_f1']:.4f}, "
                    f"Cell F1: {result['cell_em_f1']:.4f}"
                )
                f.write(
                    f"Table EM: {result['table_em']:.4f}, "
                    f"Row F1: {result['row_em_f1']:.4f}, "
                    f"Col F1: {result['col_em_f1']:.4f}, "
                    f"Cell F1: {result['cell_em_f1']:.4f}\n"
                )
            elif isinstance(result, dict) and "em" in result:
                print(
                    f"{args.dataset} Short-form EM: {result['em']:.4f}, "
                    f"Free-form F1: {result['f1']:.4f}, "
                    f"BERTScore F1: {result['bertscore_f1']:.4f}"
                )
                f.write(
                    f"Short-form EM: {result['em']:.4f}, "
                    f"Free-form F1: {result['f1']:.4f}, "
                    f"BERTScore F1: {result['bertscore_f1']:.4f}\n"
                    f"n_short_form: {result['n_short_form']}, n_free_form: {result['n_free_form']}\n"
                )
            elif isinstance(result, dict) and "f1" in result:
                print(f"{args.dataset} Test F1: {result['f1']:.4f}, CC: {result['cc']:.4f}")
                f.write(f"Test F1: {result['f1']:.4f}, CC: {result['cc']:.4f}\n")
                for k, v in result.items():
                    if k not in ("f1", "cc") and isinstance(v, dict) and "f1" in v:
                        f.write(f"  {k}: F1={v['f1']:.4f}, CC={v['cc']:.4f}\n")
            elif isinstance(result, dict):
                for k, v in result.items():
                    print(f"  {k}: {v}")
                    f.write(f"{k}: {v}\n")
            else:
                print(f"{args.dataset} Test Acc: {result}")
                f.write(f"Test Acc: {result}\n")
            f.write("\n")
    else:
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"No eval function for dataset '{args.dataset}', raw predictions saved to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.batch:
        results = run_batch_inference(args)
    else:
        results = run_inference(args)
    score_results(args, results)
