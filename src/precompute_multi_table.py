"""Precompute per-sample graph features for the FK-aware multi-table encoder (GrabMultiTable).

Launch via:
    torchrun --nproc_per_node=NUM_GPUS src/precompute_multi_table.py \\
        --dataset spider \\
        --gnn_base_model ./models/Qwen/Qwen3-Embedding-0.6B \\
        --precomputed_graphs ./precomputed/spider_multi
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

import argparse
import json
from datetime import timedelta

import torch
import torch.distributed as dist
from tqdm import tqdm

from src.dataset import load_dataset
from src.model.tokenizers import MultiTableTokenizerSplitRow
from src.utils.load_local_model import load_model_local_or_hf


def _encode_batch(base_model, batch_feats, hidden_size, chunk_size_rows):
    device = next(base_model.parameters()).device

    row_ids_list   = [f["row_input_ids"].squeeze(0) for f in batch_feats]
    row_masks_list = [f["row_attn_mask"].squeeze(0) for f in batch_feats]
    row_counts = [t.shape[0] for t in row_ids_list]
    total_rows = sum(row_counts)

    if total_rows > 0:
        cat_ids   = torch.cat(row_ids_list,   dim=0)
        cat_masks = torch.cat(row_masks_list, dim=0)
        chunks = []
        for s in range(0, total_rows, chunk_size_rows):
            e   = min(s + chunk_size_rows, total_rows)
            ids = cat_ids[s:e].to(device)
            msk = cat_masks[s:e].to(device)
            out = base_model(input_ids=ids, attention_mask=msk).last_hidden_state
            m   = msk.float().unsqueeze(-1)
            chunks.append(((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu())
        pooled     = torch.cat(chunks, dim=0)
        R_list     = list(torch.split(pooled,     row_counts, dim=0))
        rmask_list = [m.any(dim=-1) for m in torch.split(cat_masks, row_counts, dim=0)]
    else:
        R_list     = [torch.zeros((0, hidden_size)) for _ in batch_feats]
        rmask_list = [torch.zeros((0,), dtype=torch.bool) for _ in batch_feats]

    col_ids_list   = [f["col_header_ids"].squeeze(0) for f in batch_feats]
    col_masks_list = [f["col_header_mask"].squeeze(0) for f in batch_feats]
    col_counts = [t.shape[0] for t in col_ids_list]
    total_cols = sum(col_counts)

    if total_cols > 0:
        cat_col_ids   = torch.cat(col_ids_list,   dim=0).to(device)
        cat_col_masks = torch.cat(col_masks_list, dim=0).to(device)
        out = base_model(input_ids=cat_col_ids, attention_mask=cat_col_masks).last_hidden_state
        m   = cat_col_masks.float().unsqueeze(-1)
        pooled_cols = ((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu()
        C_list     = list(torch.split(pooled_cols, col_counts, dim=0))
        cmask_list = [torch.ones(c, dtype=torch.bool) for c in col_counts]
    else:
        C_list     = [torch.zeros((0, hidden_size)) for _ in batch_feats]
        cmask_list = [torch.zeros((0,), dtype=torch.bool) for _ in batch_feats]

    q_ids  = torch.cat([f["question_ids"]  for f in batch_feats], dim=0).to(device)
    q_mask = torch.cat([f["question_mask"] for f in batch_feats], dim=0).to(device)
    q_out  = base_model(input_ids=q_ids, attention_mask=q_mask).last_hidden_state.cpu()
    q_mask_cpu = q_mask.bool().cpu()

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


def _precompute_split(args, split, rank, world_size, tokenizer, base_model, hidden_size, skip_set=None):
    if skip_set is None:
        skip_set = set()
    kwargs = {}
    if args.max_rows_per_table is not None:
        kwargs['max_rows_per_table'] = args.max_rows_per_table
    ds = load_dataset[args.dataset](split, prompt_type=args.prompt_type, multi_table=True, **kwargs)
    split_dir = os.path.join(args.precomputed_graphs, split)
    os.makedirs(split_dir, exist_ok=True)
    dist.barrier()

    n = len(ds)
    all_indices = [i for i in range(rank, n, world_size) if i not in skip_set]
    pending = [idx for idx in all_indices
               if not os.path.exists(os.path.join(split_dir, f"{idx}.pt"))]

    it = tqdm(
        range(0, len(pending), args.sample_batch_size),
        desc=f"[rank {rank}] {split}",
        disable=(rank != 0),
    )

    for batch_start in it:
        batch_idx = pending[batch_start: batch_start + args.sample_batch_size]

        feats = []
        for idx in batch_idx:
            sample = ds[idx]
            dfs = sample["table"]
            question = sample["question"]
            foreign_keys = sample.get("foreign_keys", None)
            table_names = sample.get("table_names", None)
            if not isinstance(dfs, list):
                dfs = [dfs]
            feats.append(tokenizer.encode_tables(
                dfs,
                question=question,
                foreign_keys=foreign_keys,
                table_names=table_names,
            ))

        with torch.inference_mode():
            encoded = _encode_batch(base_model, feats, hidden_size, args.row_batch_size)

        for idx, feat, enc in zip(batch_idx, feats, encoded):
            out_path = os.path.join(split_dir, f"{idx}.pt")
            q_msk = enc["q_mask"]
            valid_len = int(q_msk.sum().item())
            out = {
                "R":             enc["R"].half(),
                "row_mask":      enc["row_mask"],
                "C":             enc["C"].half(),
                "col_mask":      enc["col_mask"],
                "adj":           feat["adj"].squeeze(0).contiguous(),
                "adj_cv":        feat["adj_cv"].squeeze(0).contiguous(),
                "group_to_col":  feat["group_to_col"].squeeze(0).contiguous(),
                "value_stats":   feat["value_stats"].squeeze(0).contiguous(),
                "q_tokens":      enc["q_tokens"][:valid_len].half(),
                "q_mask":        q_msk[:valid_len],
                "row_table_ids": feat["row_table_ids"].squeeze(0).contiguous(),
                "col_table_ids": feat["col_table_ids"].squeeze(0).contiguous(),
                "num_tables":    feat["num_tables"].squeeze(0),
            }
            torch.save(out, out_path)

    dist.barrier()


def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = MultiTableTokenizerSplitRow(
        base_model_name=args.gnn_base_model,
        max_length=args.question_max_len,
        num_buckets=args.num_buckets,
        max_header_len=args.max_header_len,
        max_row_len=args.row_max_len,
    )
    base_model = load_model_local_or_hf(args.gnn_base_model).to("cuda:0").eval()
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
            "multi_table":          True,
            "fk":                   True,
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
        _precompute_split(args, split, rank, world_size, tokenizer, base_model, hidden_size, skip_set=skip_set)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute multi-table graphs for GrabMultiTable")

    # Dataset
    parser.add_argument("--dataset",            type=str, required=True)
    parser.add_argument("--prompt_type",        type=str, default="qwen")
    parser.add_argument("--splits",             type=str, default="",
                        help='Comma-separated splits, e.g. "train,validation"; empty = all three')
    parser.add_argument("--max_rows_per_table", type=int, default=None,
                        help='Drop samples where any table exceeds this row count')
    parser.add_argument("--skip_list",          type=str, default="",
                        help='Path to skip_list.json')

    # GNN base model
    parser.add_argument("--gnn_base_model",     type=str, required=True)
    parser.add_argument("--question_max_len",   type=int, default=512)
    parser.add_argument("--max_header_len",     type=int, default=32)
    parser.add_argument("--row_max_len",        type=int, default=128)
    parser.add_argument("--max_columns",        type=int, default=64)
    parser.add_argument("--max_hash_groups",    type=int, default=4096)
    parser.add_argument("--num_buckets",        type=int, default=10)

    # Batching
    parser.add_argument("--row_batch_size",     type=int, default=64,
                        help='Max rows per transformer call (chunking within each sample batch)')
    parser.add_argument("--sample_batch_size",  type=int, default=32,
                        help='Number of samples to encode per GPU batch')

    # Output
    parser.add_argument("--precomputed_graphs", type=str, required=True)
    parser.add_argument("--seed",               type=int, default=42)

    args = parser.parse_args()
    main(args)
