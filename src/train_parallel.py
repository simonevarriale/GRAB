"""
Multi-GPU training with PyTorch DistributedDataParallel.

Launch via:  torchrun --nproc_per_node=NUM_GPUS src/train_parallel.py [args...]
"""
import os
import json

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

import gc
from datetime import timedelta
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_table_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything, worker_init_fn
from src.utils.lr_schedule import adjust_learning_rate
from src.global_path import precomputed_graphs_root


def is_main():
    return dist.get_rank() == 0


def _resolve_pg_path(dataset_name, prefer=None):
    """Auto-discover the precomputed graph directory for dataset_name.

    If `prefer` (a run-name, not a path) is given and exists under the dataset
    dir, use it -- this disambiguates datasets that have more than one folder.
    """
    if not precomputed_graphs_root:
        raise ValueError(
            f"$PRECOMPUTED_GRAPHS is not defined in .env — cannot auto-discover graphs for '{dataset_name}'"
        )
    dataset_dir = os.path.join(precomputed_graphs_root, dataset_name)
    if prefer:
        preferred = os.path.join(dataset_dir, prefer)
        if os.path.isdir(preferred):
            return preferred
        raise ValueError(
            f"--precomputed_graphs '{prefer}' not found under {dataset_dir}"
        )
    candidates = [
        d for d in (os.scandir(dataset_dir) if os.path.isdir(dataset_dir) else [])
        if d.is_dir()
    ]
    if len(candidates) == 1:
        return candidates[0].path
    elif len(candidates) == 0:
        raise ValueError(f"No precomputed graph folder found under {dataset_dir}")
    else:
        names = [c.name for c in candidates]
        raise ValueError(
            f"Multiple precomputed graph folders found under {dataset_dir}: {names}. "
            "Specify one with --precomputed_graphs <name>."
        )


def _build_split(ds_name, split, dataset_kwargs, skip_list_path):
    """Load one split of ds_name, optionally wrapping with SkipListDataset."""
    ds = load_dataset[ds_name](split, **dataset_kwargs)
    if skip_list_path:
        from src.dataset.precomputed_wrapper import SkipListDataset, load_skip_set
        # A dataset may index a different split's jsonl than the requested split
        # (e.g. totto_cells 'validation' is carved from the train jsonl): let it
        # redirect which skip_list.json key to use, and contribute extra indices
        # to drop (the slice belonging to the *other* split).
        skip_split = getattr(ds, 'skip_split', split)
        skip_set = set(load_skip_set(skip_list_path, ds_name, skip_split))
        skip_set |= getattr(ds, 'extra_skip_indices', set())
        ds = SkipListDataset(ds, skip_set)
    return ds


def _wrap_precomputed(ds, pg_path, split, model_name):
    """Optionally wrap ds with PrecomputedGraphDataset."""
    if pg_path and model_name in ('grab_single_table', 'grab_multi_table', 'tabert_llm'):
        from src.dataset.precomputed_wrapper import PrecomputedGraphDataset
        ds = PrecomputedGraphDataset(ds, pg_path, split)
    return ds


def _graph_row_lengths(dataset, pg_path, split):
    """Per-position length proxy (graph row count) for length-grouped batching.

    Walks the dataset wrappers (Precomputed/Tag/SkipList) to map each position to
    its original sample id, then reads the row count from the precompute's
    `_graph_sizes.json`. Returns None if the cache is absent (-> random batching).
    """
    from src.dataset.precomputed_wrapper import (
        PrecomputedGraphDataset, SkipListDataset, DatasetWithTag)

    sizes_path = os.path.join(pg_path, split, "_graph_sizes.json")
    if not os.path.isfile(sizes_path):
        return None
    with open(sizes_path) as f:
        sizes = json.load(f)

    def ordered_ids(ds):
        if isinstance(ds, (PrecomputedGraphDataset, DatasetWithTag)):
            return ordered_ids(ds.base)
        if isinstance(ds, SkipListDataset):
            base = ordered_ids(ds.base)
            return [base[i] for i in ds.valid_indices]
        return list(range(len(ds)))

    ids = ordered_ids(dataset)
    default = max((v[0] for v in sizes.values()), default=1)
    return [sizes.get(str(i), (default, 0))[0] for i in ids]


def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(0)

    seed_everything(seed=args.seed)

    if is_main():
        print(f"Configuration: {args}")
        print(f"Training with {world_size} GPUs (DDP)")

    args.llm_model_path = os.path.abspath(llama_model_path[args.llm_model_name])
    if args.table_encoder_name is None:
        if is_main():
            print("Using Baseline LLM.")
    elif args.table_encoder_name == '':
        if is_main():
            print("No table encoder specified, using default encoder untrained.")

    if not os.path.isdir(args.llm_model_path):
        raise ValueError(f"Local LLaMA model not found at: {args.llm_model_path}")

    dataset_names = [d.strip() for d in args.dataset.split(',')]
    multi_ds = len(dataset_names) > 1
    uses_graphs = args.model_name in ('grab_single_table', 'grab_multi_table', 'tabert_llm')

    multi_table = getattr(args, 'multi_table', 'True')
    dataset_kwargs = dict(prompt_type=args.prompt_type, multi_table=multi_table)
    if getattr(args, 'max_rows_per_table', None) is not None:
        dataset_kwargs['max_rows_per_table'] = args.max_rows_per_table
    if getattr(args, 'dataset_max_rows', None) is not None:
        dataset_kwargs['max_rows'] = args.dataset_max_rows

    skip_list_path = getattr(args, 'skip_list', '')

    if not multi_ds:
        # ---- Single dataset (original path) ----
        pg = getattr(args, 'precomputed_graphs', '')
        if uses_graphs:
            if not pg:
                pg = _resolve_pg_path(dataset_names[0])
                if is_main():
                    print(f"Auto-discovered precomputed graphs: {pg}")
            elif not os.path.isabs(pg):
                if not precomputed_graphs_root:
                    raise ValueError(
                        f"--precomputed_graphs '{pg}' is a relative name but "
                        "$PRECOMPUTED_GRAPHS is not set in .env"
                    )
                pg = os.path.join(precomputed_graphs_root, dataset_names[0], pg)

        train_dataset = _build_split(dataset_names[0], 'train',      dataset_kwargs, skip_list_path)
        val_dataset   = _build_split(dataset_names[0], 'validation', dataset_kwargs, skip_list_path)
        test_ds_name  = args.test_dataset if args.test_dataset else dataset_names[0]
        test_dataset  = _build_split(test_ds_name,     'test',       dataset_kwargs, skip_list_path)

        if is_main() and skip_list_path:
            print(f"After skip list: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        if getattr(args, 'filter_table_tokens', 'False') == 'True':
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
            if hasattr(train_dataset, 'filter_by_token_length'):
                train_dataset.filter_by_token_length(_tok, args.max_txt_len)
            del _tok

        # When evaluating on a different dataset than we train on (--test_dataset),
        # the test split needs ITS OWN precomputed graphs, not the train dataset's.
        test_pg = pg
        if uses_graphs and test_ds_name != dataset_names[0]:
            test_pg = _resolve_pg_path(test_ds_name)
            if is_main():
                print(f"Using test-dataset precomputed graphs for '{test_ds_name}': {test_pg}")

        train_dataset = _wrap_precomputed(train_dataset, pg,      'train',      args.model_name)
        val_dataset   = _wrap_precomputed(val_dataset,   pg,      'validation', args.model_name)
        test_dataset  = _wrap_precomputed(test_dataset,  test_pg, 'test',       args.model_name)
        if pg:
            args.precomputed_graphs = pg
            if is_main():
                print(f"Using precomputed graphs from {pg}")

        init_prompt = train_dataset.init_prompt

    else:
        # ---- Multi-dataset ----
        from torch.utils.data import ConcatDataset
        from src.dataset.precomputed_wrapper import DatasetWithTag

        # For multi-dataset runs --precomputed_graphs is a bare run-name (not a
        # path): it's applied per dataset to pick which folder when a dataset has
        # more than one. Datasets with a single folder still auto-discover.
        pg_prefer = getattr(args, 'precomputed_graphs', '') or None
        if is_main() and pg_prefer and os.path.isabs(pg_prefer):
            print("Warning: --precomputed_graphs should be a run-name (not a path) "
                  "for multi-dataset training; using it per dataset.")
        if is_main() and args.test_dataset:
            print("Warning: --test_dataset is ignored for multi-dataset training; each dataset is tested on its own test split.")

        all_train, all_val, all_test = [], [], []
        for ds_name in dataset_names:
            pg = _resolve_pg_path(ds_name, prefer=pg_prefer) if uses_graphs else None
            if is_main() and pg:
                print(f"Auto-discovered precomputed graphs for '{ds_name}': {pg}")

            train_ds = _build_split(ds_name, 'train',      dataset_kwargs, skip_list_path)
            val_ds   = _build_split(ds_name, 'validation', dataset_kwargs, skip_list_path)
            test_ds  = _build_split(ds_name, 'test',       dataset_kwargs, skip_list_path)

            train_ds = _wrap_precomputed(train_ds, pg, 'train',      args.model_name)
            val_ds   = _wrap_precomputed(val_ds,   pg, 'validation', args.model_name)
            test_ds  = _wrap_precomputed(test_ds,  pg, 'test',       args.model_name)

            if is_main():
                print(f"Dataset '{ds_name}': train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

            all_train.append(DatasetWithTag(train_ds, ds_name))
            all_val.append(DatasetWithTag(val_ds,   ds_name))
            all_test.append(DatasetWithTag(test_ds,  ds_name))

        train_dataset = ConcatDataset(all_train)
        val_dataset   = ConcatDataset(all_val)
        test_dataset  = ConcatDataset(all_test)

        if is_main():
            print(f"Multi-dataset totals: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        init_prompt = all_train[0].init_prompt

        # The model reads meta.json from args.precomputed_graphs once at __init__ to
        # get hidden_size. All datasets share the same encoder so the first path suffices.
        if uses_graphs:
            args.precomputed_graphs = _resolve_pg_path(dataset_names[0], prefer=pg_prefer)

    # Length-grouped batching (single-dataset graph runs): pack similar-length
    # tables into each batch so the GPU doesn't burn FLOPs on padding.
    train_lengths = None
    if (getattr(args, 'length_grouped', 'False') == 'True' and uses_graphs
            and not multi_ds):
        train_lengths = _graph_row_lengths(train_dataset, pg, 'train')
        if is_main():
            print(f"Length-grouped batching: {'ON' if train_lengths is not None else 'OFF (no _graph_sizes.json cache)'}")

    if train_lengths is not None:
        from src.utils.length_sampler import DistributedLengthGroupedSampler
        train_sampler = DistributedLengthGroupedSampler(
            train_lengths, args.batch_size, world_size, rank, seed=args.seed,
        )
    else:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
            seed=args.seed,
        )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        drop_last=True, pin_memory=True, collate_fn=collate_fn,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        pin_memory=True, collate_fn=collate_fn, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, sampler=test_sampler,
        drop_last=False, pin_memory=True, collate_fn=collate_fn,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )

    args.local_rank = local_rank
    model = load_model[args.model_name](init_prompt=init_prompt, args=args)

    if getattr(args, 'llm_ckpt_path', ''):
        ckpt_path = args.llm_ckpt_path
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        del checkpoint, state
        gc.collect()
        if is_main():
            print(f"Initialised from checkpoint: {ckpt_path}")
            print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model = DDP(model, device_ids=[0], find_unused_parameters=False)

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.95),
    )

    if is_main():
        trainable_params, all_param = model.module.print_trainable_params()
        print(f"Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)")
        os.makedirs(args.output_dir, exist_ok=True)

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(total=num_training_steps, desc='Training', unit='step',
                        disable=not is_main())
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        epoch_loss, accum_loss = 0.0, 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            try:
                loss = model(batch)
                loss = loss / args.grad_steps
                loss.backward()

                accum_loss += loss.item() * args.grad_steps
                epoch_loss += loss.item() * args.grad_steps
                num_batches += 1

                if (step + 1) % args.grad_steps == 0:
                    clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    adjust_learning_rate(
                        optimizer.param_groups[0], args.lr,
                        step / len(train_loader) + epoch, args,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    accum_loss = 0.0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[Rank {rank}] OOM at batch {step}, skipping. {e}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    gc.collect()
                    continue
                print(f"[Rank {rank}] Error at batch {step}: {e}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                raise

            progress_bar.set_postfix(loss=f'{loss.item() * args.grad_steps:.4f}')
            progress_bar.update(1)

        avg_train_loss = epoch_loss / max(num_batches, 1)

        val_loss_sum = 0.0
        val_steps = 0
        model.eval()
        with torch.inference_mode():
            for batch in val_loader:
                loss = model(batch)
                val_loss_sum += loss.item()
                val_steps += 1

        stats = torch.tensor([val_loss_sum, float(val_steps)], device="cuda:0")
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        val_loss = (stats[0] / stats[1]).item()

        if is_main():
            print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                _save_checkpoint(model.module, optimizer, epoch, args, is_best=True)
                print(f"New best model saved at epoch {epoch}")

        best_epoch_t = torch.tensor(best_epoch, device="cuda:0")
        dist.broadcast(best_epoch_t, src=0)
        best_epoch = int(best_epoch_t.item())

        if epoch - best_epoch >= args.patience:
            if is_main():
                print(f'Early stopping triggered! Last improvement: epoch {best_epoch}')
            break

    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    if is_main():
        print("\nStarting Evaluation...")

    eval_model = model.module
    eval_model = _reload_best_model(eval_model, args)
    eval_model.max_txt_len = args.max_txt_len
    eval_model.eval()

    local_output = []
    progress_bar_test = tqdm(range(len(test_loader)), disable=not is_main())
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = eval_model.inference(batch)
            local_output.append(output)
        progress_bar_test.update(1)

    gathered = [None] * world_size if is_main() else None
    dist.gather_object(local_output, gathered, dst=0)

    dist.barrier()
    is_rank0 = is_main()
    dist.destroy_process_group()

    if is_rank0:
        seen_ids = set()
        eval_output = []
        for shard in gathered:
            for batch_out in shard:
                batch_len = len(batch_out['id'])
                ds_tags = batch_out.get('_dataset', [None] * batch_len)
                for i in range(batch_len):
                    sid = (ds_tags[i], batch_out['id'][i])
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    eval_output.append({k: [v[i]] for k, v in batch_out.items()})

        os.makedirs(args.output_dir, exist_ok=True)

        def _report_acc(acc, path, label=''):
            tag = f' [{label}]' if label else ''
            if isinstance(acc, dict) and 'f1' in acc:
                print(f'Final Test F1{tag}: {acc["f1"]:.4f}, CC: {acc["cc"]:.4f}')
                with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{path}\nTest F1: {acc["f1"]:.4f}, CC: {acc["cc"]:.4f}\n')
            elif isinstance(acc, dict) and 'overall_acc' in acc:
                print(f'Final Test Acc{tag}: {acc["overall_acc"]:.4f}')
                with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{path}\nTest Acc: {acc["overall_acc"]:.4f}\n')
            else:
                print(f'Final Test Acc/Bleu{tag}: {acc}')
                with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{path}\nTest Acc/Bleu: {acc}\n')

        if not multi_ds:
            test_eval_name = args.test_dataset if args.test_dataset else dataset_names[0]
            path = f'{args.output_dir}/model_name_{args.model_name}_final.csv'
            acc = eval_funcs[test_eval_name](eval_output, path)
            _report_acc(acc, path)
        else:
            from collections import defaultdict
            by_ds = defaultdict(list)
            for item in eval_output:
                ds = (item.get('_dataset') or [dataset_names[0]])[0]
                by_ds[ds].append(item)
            for ds_name, ds_output in by_ds.items():
                path = f'{args.output_dir}/model_name_{args.model_name}_{ds_name}_final.csv'
                acc = eval_funcs[ds_name](ds_output, path)
                _report_acc(acc, path, label=ds_name)


if __name__ == "__main__":
    args = parse_args_table_llama()
    try:
        main(args)
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
    finally:
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
