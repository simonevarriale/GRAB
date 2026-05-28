"""Dataset wrappers:

- PrecomputedGraphDataset: attaches a precomputed graph dict to each sample.
  File path is `<precomputed_dir>/<split>/<id>.pt` where `id` is the original
  sample index. Lazy — files are loaded in DataLoader workers on __getitem__.

- SkipListDataset: filters out samples whose original indices appear in a skip
  set (e.g. samples exceeding the LLM token budget). Remaps the remaining
  samples to a contiguous index space while preserving sample['id'] so that
  PrecomputedGraphDataset still looks up the correct .pt file.

- load_skip_set: helper to read a skip_list.json for one (dataset, split).
"""

import json
import os
import torch
from torch.utils.data import Dataset


def load_skip_set(skip_list_path, dataset_name, split):
    """Return a set of original sample indices to skip for (dataset_name, split).

    Returns an empty set if the path is None, missing, or the key is absent.
    """
    if not skip_list_path or not os.path.exists(skip_list_path):
        return set()
    with open(skip_list_path) as f:
        data = json.load(f)
    indices = data.get(dataset_name, {}).get(split, {}).get("skipped_indices", [])
    return set(indices)


class SkipListDataset(Dataset):
    """Wraps a dataset and removes samples whose original indices are in skip_set."""

    def __init__(self, base_dataset, skip_set):
        self.base = base_dataset
        self.valid_indices = [i for i in range(len(base_dataset)) if i not in skip_set]
        self.init_prompt = getattr(base_dataset, 'init_prompt', None)
        if hasattr(base_dataset, 'precomputed_split'):
            self.precomputed_split = base_dataset.precomputed_split

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        return self.base[self.valid_indices[index]]


class PrecomputedGraphDataset(Dataset):
    def __init__(self, base_dataset, precomputed_dir, split):
        self.base = base_dataset
        # Datasets that carve a split from a different source split (e.g. Spider
        # validation carved from HF train) set `precomputed_split` to redirect
        # to the subdirectory where their .pt files actually live.
        precomp_split = getattr(base_dataset, 'precomputed_split', split)
        self.dir = os.path.join(precomputed_dir, precomp_split)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"Precomputed graph dir missing: {self.dir}")
        self.init_prompt = getattr(base_dataset, 'init_prompt', None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sid = sample['id']
        path = os.path.join(self.dir, f"{sid}.pt")
        try:
            sample['graph'] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        except (RuntimeError, EOFError) as e:
            raise RuntimeError(
                f"Corrupted precomputed graph file (delete and re-run precompute): {path}"
            ) from e
        return sample
