"""
Multi-GPU evaluation with PyTorch DistributedDataParallel.

Launch via:  torchrun --nproc_per_node=NUM_GPUS src/test.py [args...]
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
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.seed import seed_everything
from src.config import parse_args_table_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn


def is_main():
    return dist.get_rank() == 0


def _build_loader(dataset, batch_size, rank, world_size, num_workers=0, collator=collate_fn):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=False,
        pin_memory=True, collate_fn=collator, num_workers=num_workers,
    )


def _run_distributed_eval(model, loader, world_size, rank0):
    local_output = []
    progress_bar_test = tqdm(range(len(loader)), disable=not rank0)
    for batch in loader:
        with torch.inference_mode():
            output = model.inference(batch)
            local_output.append(output)
        progress_bar_test.update(1)

    gathered = [None] * world_size if rank0 else None
    dist.gather_object(local_output, gathered, dst=0)
    return gathered


def _flatten_and_dedupe(gathered):
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
    return eval_output


def _score_and_log(eval_output, dataset_name, args, seed):
    os.makedirs(f'{args.output_dir}/{dataset_name}', exist_ok=True)
    path = (
        f'{args.output_dir}/{dataset_name}/bmix_model_name_{args.model_name}'
        f'_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}'
        f'_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}'
        f'_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    )
    result = eval_funcs[dataset_name](eval_output, path)
    if isinstance(result, dict) and 'overall_acc' in result:
        print(f'{dataset_name} Test Acc: {result["overall_acc"]:.4f}')
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(f'{path}\n')
            for key, val in result.items():
                if key == 'overall_acc':
                    file.write(f'Overall Acc: {val:.4f}\n')
                else:
                    file.write(f'{key}: acc={val["acc"]:.4f}, count={val["count"]}\n')
            file.write('\n')
    elif isinstance(result, dict) and 'table_em' in result:
        print(
            f'{dataset_name} Table EM: {result["table_em"]:.4f}, '
            f'Row F1: {result["row_em_f1"]:.4f}, '
            f'Col F1: {result["col_em_f1"]:.4f}, '
            f'Cell F1: {result["cell_em_f1"]:.4f}'
        )
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(
                f'{path} \nTable EM: {result["table_em"]:.4f}, '
                f'Row F1: {result["row_em_f1"]:.4f}, '
                f'Col F1: {result["col_em_f1"]:.4f}, '
                f'Cell F1: {result["cell_em_f1"]:.4f}\n'
            )
    elif isinstance(result, dict) and 'em' in result and 'f1' not in result:
        print(f'{dataset_name} Test EM: {result["em"]:.4f}')
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(f'{path} \nTest EM: {result["em"]:.4f}\n')
    elif isinstance(result, dict) and 'em' in result:
        print(
            f'{dataset_name} Short-form EM: {result["em"]:.4f}, '
            f'Free-form F1: {result["f1"]:.4f}, '
            f'BERTScore F1: {result["bertscore_f1"]:.4f}'
        )
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(
                f'{path} \nShort-form EM: {result["em"]:.4f}, '
                f'Free-form F1: {result["f1"]:.4f}, '
                f'BERTScore F1: {result["bertscore_f1"]:.4f}\n'
                f'n_short_form: {result["n_short_form"]}, n_free_form: {result["n_free_form"]}\n'
            )
    elif isinstance(result, dict):
        print(f'{dataset_name} Test F1: {result["f1"]:.4f}, CC: {result["cc"]:.4f}')
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(f'{path} \nTest F1: {result["f1"]:.4f}, CC: {result["cc"]:.4f}\n')
    else:
        print(f'{dataset_name} Test Acc/blue: {result}')
        with open(f'{args.output_dir}/{dataset_name}/score.txt', 'a', encoding='utf-8') as file:
            file.write(f'{path} \nTest Acc/blue: {result}\n')


def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(0)

    seed = args.seed
    seed_everything(seed=seed)

    if is_main():
        print(args)
        print(f"Evaluating with {world_size} GPUs (DDP)")

    multi_table = getattr(args, 'multi_table', 'True')

    def _make_dataset_kwargs():
        kwargs = dict(prompt_type=args.prompt_type, multi_table=multi_table)
        if getattr(args, 'max_rows_per_table', None) is not None:
            kwargs['max_rows_per_table'] = args.max_rows_per_table
        if getattr(args, 'dataset_max_rows', None) is not None:
            kwargs['max_rows'] = args.dataset_max_rows
        return kwargs

    test_dataset = load_dataset[args.dataset]('test', **_make_dataset_kwargs())
    init_prompt = test_dataset.init_prompt

    second_test_dataset = None
    if args.second_dataset != '':
        second_test_dataset = load_dataset[args.second_dataset]('test', **_make_dataset_kwargs())

    if getattr(args, 'skip_list', ''):
        from src.dataset.precomputed_wrapper import SkipListDataset, load_skip_set
        test_dataset = SkipListDataset(test_dataset, load_skip_set(args.skip_list, args.dataset, 'test'))
        if second_test_dataset is not None:
            second_test_dataset = SkipListDataset(
                second_test_dataset, load_skip_set(args.skip_list, args.second_dataset, 'test')
            )

    if is_main():
        print(f"Test dataset size: {len(test_dataset)}")

    if getattr(args, 'precomputed_graphs', '') and args.model_name in ('grab_single_table', 'grab_multi_table'):
        from src.dataset.precomputed_wrapper import PrecomputedGraphDataset
        test_dataset = PrecomputedGraphDataset(test_dataset, args.precomputed_graphs, 'test')
        if second_test_dataset is not None:
            second_test_dataset = PrecomputedGraphDataset(
                second_test_dataset, args.precomputed_graphs, 'test'
            )
        if is_main():
            print(f"Using precomputed graphs from {args.precomputed_graphs}")

    raw_path = llama_model_path[args.llm_model_name]
    args.llm_model_path = os.path.abspath(raw_path)

    if is_main():
        if args.table_encoder_name is None:
            print("Using Baseline LLM.")
        elif args.table_encoder_name == '':
            print("No table encoder specified, using default encoder untrained.")

    if not os.path.isdir(args.llm_model_path):
        raise ValueError(f"LLM model not found at: {args.llm_model_path}")

    model = load_model[args.model_name](init_prompt=init_prompt, args=args)

    if os.path.isdir(args.llm_ckpt_path) and os.path.abspath(args.llm_ckpt_path) != args.llm_model_path:
        if is_main():
            print(f"Loading LLM from HuggingFace folder: {args.llm_ckpt_path}")
        args.llm_model_path = os.path.abspath(args.llm_ckpt_path)
        model = load_model[args.model_name](init_prompt=init_prompt, args=args)

    elif os.path.exists(args.llm_ckpt_path):
        checkpoint = torch.load(args.llm_ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            ckpt_args = checkpoint["config"]
            for attr in ('gnn_base_model', 'num_gnn_layers', 'num_latents',
                         'num_row_latents', 'num_col_latents', 'num_val_latents',
                         'num_resampler_heads', 'num_resampler_layers',
                         'max_columns', 'max_hash_groups',
                         'max_header_len', 'projector_type'):
                if hasattr(ckpt_args, attr):
                    old_val = getattr(args, attr, None)
                    new_val = getattr(ckpt_args, attr)
                    if old_val != new_val and is_main():
                        print(f"Overriding {attr}: {old_val} -> {new_val} (from checkpoint)")
                    setattr(args, attr, new_val)

            model = load_model[args.model_name](init_prompt=init_prompt, args=args)

        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main():
            print(f'load from ckpt {args.llm_ckpt_path}')
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                print("Example missing keys:", missing[:20])
            if len(unexpected) > 0:
                print("Example unexpected keys:", unexpected[:20])

        del checkpoint
        del state
        gc.collect()
    elif args.llm_ckpt_path:
        raise ValueError(f"Checkpoint not found: {args.llm_ckpt_path}")

    if not os.path.isdir(args.llm_model_path):
        raise ValueError(f"LLM model not found at: {args.llm_model_path}")

    model.eval()

    test_loader = _build_loader(
        test_dataset, args.eval_batch_size, rank, world_size, args.num_workers,
    )
    second_test_loader = None
    if second_test_dataset is not None:
        second_test_loader = _build_loader(
            second_test_dataset, 4, rank, world_size, args.num_workers,
        )

    dist.barrier()
    if is_main():
        print("\nStarting Evaluation...")

    gathered = _run_distributed_eval(model, test_loader, world_size, is_main())

    gathered_second = None
    if second_test_loader is not None:
        if is_main():
            print("\nStarting Evaluation on second dataset...")
        gathered_second = _run_distributed_eval(
            model, second_test_loader, world_size, is_main()
        )

    dist.barrier()
    is_rank0 = is_main()
    dist.destroy_process_group()

    if is_rank0:
        eval_output = _flatten_and_dedupe(gathered)
        _score_and_log(eval_output, args.dataset, args, seed)

        if gathered_second is not None:
            eval_output_second = _flatten_and_dedupe(gathered_second)
            _score_and_log(eval_output_second, args.second_dataset, args, seed)



if __name__ == "__main__":
    args = parse_args_table_llama()
    try:
        main(args)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise e
    finally:
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()
