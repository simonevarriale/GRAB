import os
import torch


def _ckpt_tag(args):
    if getattr(args, 'llm_frozen', 'False') == 'True':
        llm_mode = "llm_frozen"
    elif getattr(args, 'llm_lora', 'False') == 'True':
        llm_mode = "llm_lora"
    else:
        llm_mode = "llm_trained"
    enc_mode = "encoder_frozen" if getattr(args, 'table_encoder_frozen', 'False') == 'True' else "encoder_trained"
    return (
        f"{args.dataset}"
        f"__{args.llm_model_name}"
        f"__{llm_mode}"
        f"__gnn_layers_{getattr(args, 'num_gnn_layers', '?')}"
        f"__row_latents_{getattr(args, 'num_row_latents', '?')}"
        f"__col_latents_{getattr(args, 'num_col_latents', '?')}"
        f"__val_latents_{getattr(args, 'num_val_latents', '?')}"
        f"__resampler_layers_{getattr(args, 'num_resampler_layers', '?')}"
        f"__max_columns_{getattr(args, 'max_columns', '?')}"
        f"__max_hash_groups_{getattr(args, 'max_hash_groups', '?')}"
        f"__{enc_mode}"
        f"__lr_{args.lr}"
        f"__seed_{args.seed}"
    )


def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    tag = _ckpt_tag(args)
    suffix = "best" if is_best else str(cur_epoch)
    path = f'{args.output_dir}/{tag}_checkpoint_{suffix}.pth'
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    tag = _ckpt_tag(args)
    checkpoint_path = f'{args.output_dir}/{tag}_checkpoint_best.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
