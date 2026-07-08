from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from src.global_path import model_dir


def load_tokenizer_local_or_hf(model_name: str):
    local_dir = Path(model_dir) / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        print(f"Loaded tokenizer from local: {local_dir}")
    except Exception:
        print(f"Tokenizer not found locally. Downloading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        print(f"Saved tokenizer to: {local_dir}")

    return tokenizer


def load_model_local_or_hf(model_name: str, torch_dtype=None):
    import torch
    local_dir = Path(model_dir) / model_name
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"dtype": torch_dtype} if torch_dtype is not None else {}

    try:
        model = AutoModel.from_pretrained(local_dir, local_files_only=True, **kwargs)
        print(f"Loaded model from local: {local_dir}")
    except Exception:
        print(f"Model not found locally. Downloading: {model_name}")
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.save_pretrained(local_dir)
        print(f"Saved model to: {local_dir}")

    return model