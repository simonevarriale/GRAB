"""Precompute per-sample graph features for the single-table row encoder (GrabSingleTable).

Launch via:
    torchrun --nproc_per_node=NUM_GPUS src/precompute_single_table.py \\
        --dataset wtq \\
        --gnn_base_model ./models/Qwen/Qwen3-Embedding-0.6B \\
        --precomputed_graphs ./precomputed/wtq_single
"""

import os

_local_rank = int(os.environ.get("LOCAL_RANK", 0))
_parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _parent_visible:
    _visible_list = [d.strip() for d in _parent_visible.split(",") if d.strip()]
    if _local_rank >= len(_visible_list):
        raise RuntimeError(
            f"LOCAL_RANK={_local_rank} but CUDA_VISIBLE_DEVICES only exposes "
            f"{len(_visible_list)} device(s): {_visible_list}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible_list[_local_rank]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

# Each prefetch worker tokenizes independently; keep the Rust tokenizer
# single-threaded so N workers use N cores predictably instead of oversubscribing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from src.dataset import load_dataset
from src.model.tokenizers import TableTokenizerRow
from src.utils.load_local_model import load_model_local_or_hf

# HF fast tokenizers are NOT thread-safe (shared truncation/padding state ->
# "Already borrowed"), so each prefetch worker builds its own instance.
_TLS = threading.local()
_TOK_LOCK = threading.Lock()


def _get_tokenizer(tok_kwargs):
    tok = getattr(_TLS, "tok", None)
    if tok is None:
        with _TOK_LOCK:  # serialise construction; concurrent use after is fine
            tok = TableTokenizerRow(**tok_kwargs)
        _TLS.tok = tok
    return tok


def _forward_pooled(base_model, cat_ids, cat_masks, chunk_size):
    """Chunked encoder forward with mean pooling.

    Each chunk is trimmed to its longest real sequence so the fixed
    max_length padding from the tokenizer is not paid in compute.
    """
    device = next(base_model.parameters()).device
    chunks = []
    for s in range(0, cat_ids.shape[0], chunk_size):
        msk_cpu = cat_masks[s:s + chunk_size]
        L   = max(int(msk_cpu.sum(dim=1).max().item()), 1)
        ids = cat_ids[s:s + chunk_size, :L].to(device)
        msk = msk_cpu[:, :L].to(device)
        out = base_model(input_ids=ids, attention_mask=msk).last_hidden_state.float()
        m   = msk.float().unsqueeze(-1)
        chunks.append(((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu())
    return torch.cat(chunks, dim=0)


def _encode_batch(base_model, batch_feats, batch_q_ids, batch_q_masks, hidden_size, chunk_size_rows):
    device = next(base_model.parameters()).device

    row_ids_list   = [f["row_input_ids"].squeeze(0)      for f in batch_feats]
    row_masks_list = [f["row_attention_mask"].squeeze(0) for f in batch_feats]
    row_counts = [t.shape[0] for t in row_ids_list]
    total_rows = sum(row_counts)

    if total_rows > 0:
        cat_masks_cpu = torch.cat(row_masks_list, dim=0)
        cat_ids_cpu   = torch.cat(row_ids_list,   dim=0)
        pooled     = _forward_pooled(base_model, cat_ids_cpu, cat_masks_cpu, chunk_size_rows)
        R_list     = list(torch.split(pooled,         row_counts, dim=0))
        rmask_list = [m.any(dim=-1) for m in torch.split(cat_masks_cpu, row_counts, dim=0)]
    else:
        R_list     = [torch.zeros((0, hidden_size)) for _ in batch_feats]
        rmask_list = [torch.zeros((0,), dtype=torch.bool) for _ in batch_feats]

    col_ids_list   = [f["col_header_ids"].squeeze(0)  for f in batch_feats]
    col_masks_list = [f["col_header_mask"].squeeze(0) for f in batch_feats]
    col_counts = [t.shape[0] for t in col_ids_list]
    total_cols = sum(col_counts)

    if total_cols > 0:
        pooled_cols = _forward_pooled(
            base_model,
            torch.cat(col_ids_list, dim=0),
            torch.cat(col_masks_list, dim=0),
            chunk_size_rows,
        )
        C_list     = list(torch.split(pooled_cols, col_counts, dim=0))
        cmask_list = [torch.ones(c, dtype=torch.bool) for c in col_counts]
    else:
        C_list     = [torch.zeros((0, hidden_size)) for _ in batch_feats]
        cmask_list = [torch.zeros((0,), dtype=torch.bool) for _ in batch_feats]

    q_ids_cpu  = torch.cat(batch_q_ids,   dim=0)
    q_mask_cpu = torch.cat(batch_q_masks, dim=0)
    Lq = max(int(q_mask_cpu.sum(dim=1).max().item()), 1)
    q_ids  = q_ids_cpu[:, :Lq].to(device)
    q_mask = q_mask_cpu[:, :Lq].to(device)
    q_out  = base_model(input_ids=q_ids, attention_mask=q_mask).last_hidden_state.float().cpu()
    q_mask_cpu = q_mask_cpu[:, :Lq].bool()

    return [
        {
            "R":        R_list[i].contiguous(),
            "row_mask": rmask_list[i].contiguous(),
            "C":        C_list[i].contiguous(),
            "col_mask": cmask_list[i].contiguous(),
            "q_tokens": q_out[i].contiguous(),
            "q_mask":   q_mask_cpu[i].contiguous(),
        }
        for i in range(len(batch_feats))
    ]


def _is_oversized(feat, q_attention_mask):
    if feat["row_attention_mask"].squeeze(0)[:, -1].any():
        return True
    if feat["col_header_mask"].squeeze(0)[:, -1].any():
        return True
    if q_attention_mask[0, -1].item() == 1:
        return True
    return False


def _prepare_batch_cpu(ds, batch_idx, tok_kwargs):
    tokenizer = _get_tokenizer(tok_kwargs)  # thread-local instance
    feats, q_ids_list, q_masks_list, valid_idx, excluded = [], [], [], [], []
    for idx in batch_idx:
        sample = ds[idx]
        table = sample["table"]
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)
        if table.empty:
            excluded.append(idx)
            continue
        feat = tokenizer.encode_table(table.astype(str))
        enc = tokenizer.tokenizer(
            sample["question"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.max_length,
            return_tensors="pt",
        )
        if _is_oversized(feat, enc["attention_mask"]):
            excluded.append(idx)
            continue
        feats.append(feat)
        q_ids_list.append(enc["input_ids"])
        q_masks_list.append(enc["attention_mask"])
        valid_idx.append(idx)
    return feats, q_ids_list, q_masks_list, valid_idx, excluded


def _precompute_split(args, split, rank, world_size, tok_kwargs, base_model, hidden_size, skip_set=None):
    if skip_set is None:
        skip_set = set()
    kwargs = dict(prompt_type=args.prompt_type)
    if getattr(args, 'dataset_max_rows', None) is not None:
        kwargs['max_rows'] = args.dataset_max_rows
    try:
        ds = load_dataset[args.dataset](split, multi_table=False, **kwargs)
    except TypeError as e:
        if "multi_table" not in str(e):
            raise
        ds = load_dataset[args.dataset](split, **kwargs)
    split_dir = os.path.join(args.precomputed_graphs, split)
    os.makedirs(split_dir, exist_ok=True)
    dist.barrier()

    n = len(ds)
    all_indices = [i for i in range(rank, n, world_size) if i not in skip_set]
    pending = [idx for idx in all_indices
               if not os.path.exists(os.path.join(split_dir, f"{idx}.pt"))]

    batches = [pending[s:s + args.sample_batch_size]
               for s in range(0, len(pending), args.sample_batch_size)]

    it = tqdm(total=len(batches), desc=f"[rank {rank}] {split}", disable=(rank != 0))
    excluded_indices = []

    # Multi-worker prefetch: several CPU workers tokenize batches ahead while the
    # (single) GPU encodes. Keep a few batches in flight so workers never idle.
    n_workers = max(1, getattr(args, "prefetch_workers", 4))
    max_depth = n_workers + 2
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        inflight = deque()
        bi = 0

        def fill():
            nonlocal bi
            while len(inflight) < max_depth and bi < len(batches):
                inflight.append(pool.submit(_prepare_batch_cpu, ds, batches[bi], tok_kwargs))
                bi += 1

        fill()
        while inflight:
            feats, q_ids_list, q_masks_list, valid_idx, excluded_batch = inflight.popleft().result()
            fill()  # top back up so workers stay busy during the GPU forward
            excluded_indices.extend(excluded_batch)

            if not feats:
                it.update(1)
                continue

            try:
                with torch.inference_mode():
                    encoded = _encode_batch(base_model, feats, q_ids_list, q_masks_list,
                                            hidden_size, args.row_batch_size)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[rank {rank}] OOM on batch {valid_idx}, skipping {len(valid_idx)} samples.")
                excluded_indices.extend(valid_idx)
                it.update(1)
                continue

            for idx, feat, enc in zip(valid_idx, feats, encoded):
                out_path = os.path.join(split_dir, f"{idx}.pt")
                q_msk = enc["q_mask"]
                valid_len = int(q_msk.sum().item())
                out = {
                    "R":            enc["R"].half(),
                    "row_mask":     enc["row_mask"],
                    "C":            enc["C"].half(),
                    "col_mask":     enc["col_mask"],
                    "q_tokens":     enc["q_tokens"][:valid_len].half(),
                    "q_mask":       q_msk[:valid_len],
                    "adj":          feat["adj"].squeeze(0).contiguous(),
                    "adj_cv":       feat["adj_cv"].squeeze(0).contiguous(),
                    "group_to_col": feat["group_to_col"].squeeze(0).contiguous(),
                    "value_stats":  feat["value_stats"].squeeze(0).contiguous(),
                }
                # Legacy (non-zip) format: torch's zip reader (PyTorchFileReader)
                # raises OSError [Errno 22] under concurrent DataLoader workers on
                # networked scratch. Legacy files use a robust sequential reader.
                torch.save(out, out_path, _use_new_zipfile_serialization=False)

            it.update(1)
    it.close()

    rank_excl_path = os.path.join(split_dir, f"excluded_rank{rank}.json")
    with open(rank_excl_path, "w") as f:
        json.dump(excluded_indices, f)
    dist.barrier()

    if rank == 0:
        all_excluded = []
        for r in range(world_size):
            p = os.path.join(split_dir, f"excluded_rank{r}.json")
            with open(p) as f:
                all_excluded.extend(json.load(f))
            os.remove(p)
        all_excluded = sorted(set(all_excluded))
        excl_path = os.path.join(split_dir, "excluded.json")
        with open(excl_path, "w") as f:
            json.dump({"count": len(all_excluded), "indices": all_excluded}, f, indent=2)
        print(f"  [{split}] Excluded {len(all_excluded)} / {n} oversized samples → {excl_path}")

    dist.barrier()


def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    tok_kwargs = dict(
        base_model_name=args.gnn_base_model,
        max_length=args.question_max_len,
        num_buckets=args.num_buckets,
        max_header_len=args.max_header_len,
        row_max_len=args.row_max_len,
    )
    base_model = load_model_local_or_hf(args.gnn_base_model, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    hidden_size = base_model.config.hidden_size

    os.makedirs(args.precomputed_graphs, exist_ok=True)
    if rank == 0:
        meta = {
            "dataset":              args.dataset,
            "prompt_type":          args.prompt_type,
            "gnn_base_model":       args.gnn_base_model,
            "hidden_size":          int(hidden_size),
            "question_max_len":     args.question_max_len,
            "max_header_len":       args.max_header_len,
            "row_max_len":          args.row_max_len,
            "row_batch_size":       args.row_batch_size,
            "num_buckets":          args.num_buckets,
            "seed":                 args.seed,
            "multi_table":          False,
            "row_variant":          True,
            "question_conditioned": True,
        }
        with open(os.path.join(args.precomputed_graphs, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    dist.barrier()

    skip_data = {}
    if getattr(args, 'skip_list', '') and os.path.exists(args.skip_list):
        with open(args.skip_list) as f:
            skip_data = json.load(f)
        if rank == 0:
            print(f"Loaded skip list from {args.skip_list}")

    splits = args.splits.split(",") if args.splits else ["train", "validation", "test"]
    for split in splits:
        if rank == 0:
            print(f"\n=== Precomputing split: {split} ===")
        skip_set = set(skip_data.get(args.dataset, {}).get(split, {}).get('skipped_indices', []))
        if rank == 0 and skip_set:
            print(f"  Skipping {len(skip_set)} samples from skip list.")
        _precompute_split(args, split, rank, world_size, tok_kwargs, base_model, hidden_size, skip_set=skip_set)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute single-table graphs for GrabSingleTable")

    # Dataset
    parser.add_argument("--dataset",           type=str, required=True)
    parser.add_argument("--prompt_type",       type=str, default="qwen")
    parser.add_argument("--splits",            type=str, default="",
                        help='Comma-separated splits, e.g. "train,validation"; empty = all three')
    parser.add_argument("--dataset_max_rows",  type=int, default=None,
                        help='Truncate each table to this many rows before encoding')
    parser.add_argument("--skip_list",         type=str, default="",
                        help='Path to skip_list.json')

    # GNN base model
    parser.add_argument("--gnn_base_model",    type=str, required=True)
    parser.add_argument("--question_max_len",  type=int, default=512)
    parser.add_argument("--max_header_len",    type=int, default=32)
    parser.add_argument("--row_max_len",       type=int, default=128)
    parser.add_argument("--max_columns",       type=int, default=64)
    parser.add_argument("--max_hash_groups",   type=int, default=4096)
    parser.add_argument("--num_buckets",       type=int, default=10)

    # Batching
    parser.add_argument("--row_batch_size",    type=int, default=64,
                        help='Max rows per transformer call (chunking within each sample batch)')
    parser.add_argument("--sample_batch_size", type=int, default=32,
                        help='Number of samples to encode per GPU batch')
    parser.add_argument("--prefetch_workers",  type=int, default=4,
                        help='CPU threads tokenizing batches ahead of the GPU')

    # Output
    parser.add_argument("--precomputed_graphs", type=str, required=True)
    parser.add_argument("--seed",               type=int, default=42)

    args = parser.parse_args()
    main(args)
