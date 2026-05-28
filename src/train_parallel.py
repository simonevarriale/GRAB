"""
Multi-GPU training with PyTorch DistributedDataParallel.

Launch via:  torchrun --nproc_per_node=NUM_GPUS src/train_parallel.py [args...]
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

    if args.model_name in ('grab_single_table', 'grab_multi_table'):
        pg = getattr(args, 'precomputed_graphs', '')
        if not pg:
            # Auto-discover: find the single subdirectory under $PRECOMPUTED_GRAPHS/<dataset>/
            if not precomputed_graphs_root:
                raise ValueError(
                    "--precomputed_graphs was not set and $PRECOMPUTED_GRAPHS is not defined in .env"
                )
            dataset_dir = os.path.join(precomputed_graphs_root, args.dataset)
            candidates = [
                d for d in (os.scandir(dataset_dir) if os.path.isdir(dataset_dir) else [])
                if d.is_dir()
            ]
            if len(candidates) == 1:
                args.precomputed_graphs = candidates[0].path
                if is_main():
                    print(f"Auto-discovered precomputed graphs: {args.precomputed_graphs}")
            elif len(candidates) == 0:
                raise ValueError(f"No precomputed graph folder found under {dataset_dir}")
            else:
                names = [c.name for c in candidates]
                raise ValueError(
                    f"Multiple precomputed graph folders found under {dataset_dir}: {names}. "
                    "Specify one with --precomputed_graphs <name>."
                )
        elif not os.path.isabs(pg):
            if not precomputed_graphs_root:
                raise ValueError(
                    f"--precomputed_graphs '{pg}' is a relative name but "
                    "$PRECOMPUTED_GRAPHS is not set in .env"
                )
            args.precomputed_graphs = os.path.join(precomputed_graphs_root, args.dataset, pg)

    args.llm_model_path = os.path.abspath(llama_model_path[args.llm_model_name])
    if args.table_encoder_name is None:
        if is_main():
            print("Using Baseline LLM.")
    elif args.table_encoder_name == '':
        if is_main():
            print("No table encoder specified, using default encoder untrained.")

    if not os.path.isdir(args.llm_model_path):
        raise ValueError(f"Local LLaMA model not found at: {args.llm_model_path}")

    multi_table = getattr(args, 'multi_table', 'True')
    dataset_kwargs = dict(prompt_type=args.prompt_type, multi_table=multi_table)
    if getattr(args, 'max_rows_per_table', None) is not None:
        dataset_kwargs['max_rows_per_table'] = args.max_rows_per_table
    if getattr(args, 'dataset_max_rows', None) is not None:
        dataset_kwargs['max_rows'] = args.dataset_max_rows

    train_dataset = load_dataset[args.dataset]('train', **dataset_kwargs)
    val_dataset   = load_dataset[args.dataset]('validation', **dataset_kwargs)
    if args.test_dataset == '':
        test_dataset = load_dataset[args.dataset]('test', **dataset_kwargs)
    else:
        test_dataset = load_dataset[args.test_dataset]('test', **dataset_kwargs)

    if getattr(args, 'skip_list', ''):
        from src.dataset.precomputed_wrapper import SkipListDataset, load_skip_set
        train_dataset = SkipListDataset(train_dataset, load_skip_set(args.skip_list, args.dataset, 'train'))
        val_dataset   = SkipListDataset(val_dataset,   load_skip_set(args.skip_list, args.dataset, 'validation'))
        test_ds_name  = args.test_dataset if args.test_dataset else args.dataset
        test_dataset  = SkipListDataset(test_dataset,  load_skip_set(args.skip_list, test_ds_name, 'test'))
        if is_main():
            print(f"After skip list: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    if getattr(args, 'filter_table_tokens', 'False') == 'True':
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
        if hasattr(train_dataset, 'filter_by_token_length'):
            train_dataset.filter_by_token_length(_tok, args.max_txt_len)
        del _tok

    if getattr(args, 'precomputed_graphs', '') and args.model_name in ('grab_single_table', 'grab_multi_table'):
        from src.dataset.precomputed_wrapper import PrecomputedGraphDataset
        train_dataset = PrecomputedGraphDataset(train_dataset, args.precomputed_graphs, 'train')
        val_dataset   = PrecomputedGraphDataset(val_dataset,   args.precomputed_graphs, 'validation')
        test_dataset  = PrecomputedGraphDataset(test_dataset,  args.precomputed_graphs, 'test')
        if is_main():
            print(f"Using precomputed graphs from {args.precomputed_graphs}")

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
        pin_memory=True, collate_fn=collate_fn, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn, persistent_workers=persistent,
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
    model = load_model[args.model_name](init_prompt=train_dataset.init_prompt, args=args)
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
                for i in range(batch_len):
                    sid = batch_out['id'][i]
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    eval_output.append({k: [v[i]] for k, v in batch_out.items()})

        path = f'{args.output_dir}/model_name_{args.model_name}_final.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        acc = eval_funcs[args.dataset](eval_output, path)
        if isinstance(acc, dict) and 'f1' in acc:
            print(f'Final Test F1: {acc["f1"]:.4f}, CC: {acc["cc"]:.4f}')
            with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as file:
                file.write(f'{path} \nTest F1: {acc["f1"]:.4f}, CC: {acc["cc"]:.4f}\n')
        elif isinstance(acc, dict) and 'overall_acc' in acc:
            print(f'Final Test Acc: {acc["overall_acc"]:.4f}')
            with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as file:
                file.write(f'{path} \nTest Acc: {acc["overall_acc"]:.4f}\n')
        else:
            print(f'Final Test Acc/Blue: {acc}')
            with open(f'{args.output_dir}/score.txt', 'a', encoding='utf-8') as file:
                file.write(f'{path} \nTest Acc/Blue: {acc}\n')


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
