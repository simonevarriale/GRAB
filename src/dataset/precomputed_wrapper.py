"""Dataset wrappers:

- DatasetWithTag: stamps a '_dataset' key onto every sample so that combined
  datasets can be routed to the correct eval function after inference.

- PrecomputedGraphDataset: attaches a precomputed graph dict to each sample.
  File path is `<precomputed_dir>/<split>/<id>.pt` where `id` is the original
  sample index. Lazy — files are loaded in DataLoader workers on __getitem__.

- SkipListDataset: filters out samples whose original indices appear in a skip
  set (e.g. samples exceeding the LLM token budget). Remaps the remaining
  samples to a contiguous index space while preserving sample['id'] so that
  PrecomputedGraphDataset still looks up the correct .pt file.

- load_skip_set: helper to read a skip_list.json for one (dataset, split).
"""

import glob
import json
import os
import torch
from torch.utils.data import Dataset


# Graph fields kept for self-supervised structural pretraining. q_tokens/q_mask
# are dropped (the question is irrelevant to structure) so that single- and
# multi-table graphs collate into a uniform batch.
_PRETRAIN_GRAPH_KEYS = (
    "R", "row_mask", "C", "col_mask",
    "adj", "adj_cv", "group_to_col", "value_stats",
)


class GraphOnlyDataset(Dataset):
    """Pools precomputed graph .pt files for self-supervised GNN pretraining.

    Needs no base text dataset — globs `<id>.pt` files directly from one or more
    precomputed split directories. Single-table graphs are normalised to carry
    zero table-ids (num_tables=1) so they share a batch with multi-table graphs,
    and only the structural fields in `_PRETRAIN_GRAPH_KEYS` are returned.
    """

    def __init__(self, dirs, load_tapas=False, load_totto=False,
                 max_rows=None, max_groups=None):
        # load_tapas: attach question conditioning (q_tokens/q_mask) + TAPAS-style
        # supervision (col_relevant/agg_op) from the parallel "<split>_tapas" dir
        # (built by src/build_tapas_labels.py) for Stage-2 resampler training.
        # load_totto: attach q_tokens + cell-grounding supervision (cell_target)
        # from "<split>_totto" (built by src/build_totto_labels.py).
        # max_rows / max_groups: drop pathologically large graphs (a few huge
        # tables blow up the padded [B, rows, groups] collate tensors -> GPU OOM);
        # needs a `_graph_sizes.json` cache (src/scan_graph_sizes.py) per dir.
        self.load_tapas = load_tapas
        self.load_totto = load_totto
        self.files = []
        n_dropped = 0
        for d in dirs:
            files = sorted(glob.glob(os.path.join(d, "*.pt")))
            if max_rows or max_groups:
                sizes_path = os.path.join(d, "_graph_sizes.json")
                if not os.path.isfile(sizes_path):
                    raise FileNotFoundError(
                        f"--max_rows/--max_groups need a size cache; run "
                        f"src/scan_graph_sizes.py to create {sizes_path}")
                with open(sizes_path) as f:
                    sizes = json.load(f)
                kept = []
                for p in files:
                    gid = os.path.basename(p)[:-3]
                    rg = sizes.get(gid)
                    if rg is None:
                        kept.append(p)  # unknown size -> keep
                        continue
                    r, g = rg
                    if (max_rows and r > max_rows) or (max_groups and g > max_groups):
                        n_dropped += 1
                        continue
                    kept.append(p)
                files = kept
            self.files.extend(files)
        if not self.files:
            raise FileNotFoundError(f"No .pt graph files found under: {list(dirs)}")
        self.n_dropped = n_dropped

    def __len__(self):
        return len(self.files)

    def _sidecar_path(self, graph_path, suffix):
        d, fn = os.path.split(graph_path)
        return os.path.join(d + suffix, fn)

    def __getitem__(self, index):
        path = self.files[index]
        # Graphs are saved in legacy (non-zip) format because torch's zip reader
        # (PyTorchFileReader) crashes under concurrent DataLoader workers on
        # networked scratch; legacy files don't support mmap, so mmap=False.
        raw = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
        g = {k: raw[k] for k in _PRETRAIN_GRAPH_KEYS if k in raw}
        if self.load_tapas:
            if "q_tokens" in raw:
                g["q_tokens"] = raw["q_tokens"]
                g["q_mask"] = raw["q_mask"]
            n_c = g["C"].shape[0]
            tp = self._sidecar_path(path, "_tapas")
            if os.path.exists(tp):
                t = torch.load(tp, map_location='cpu', weights_only=False)
                g["col_relevant"] = t["col_relevant"]
                g["agg_op"] = t["agg_op"]
                g["tapas_valid"] = t["tapas_valid"]
            else:
                g["col_relevant"] = torch.zeros(n_c, dtype=torch.bool)
                g["agg_op"] = torch.tensor(0, dtype=torch.long)
                g["tapas_valid"] = torch.tensor(False)
        if self.load_totto:
            if "q_tokens" in raw:
                g["q_tokens"] = raw["q_tokens"]
                g["q_mask"] = raw["q_mask"]
            tp = self._sidecar_path(path, "_totto")
            if os.path.exists(tp):
                t = torch.load(tp, map_location='cpu', weights_only=False)
                g["cell_target"] = t["cell_target"]
                g["totto_valid"] = t["totto_valid"]
            else:
                g["cell_target"] = torch.zeros_like(g["adj"])
                g["totto_valid"] = torch.tensor(False)
        if "row_table_ids" in raw:
            g["row_table_ids"] = raw["row_table_ids"]
            g["col_table_ids"] = raw["col_table_ids"]
            g["num_tables"] = raw["num_tables"]
        else:
            g["row_table_ids"] = torch.zeros(g["R"].shape[0], dtype=torch.long)
            g["col_table_ids"] = torch.zeros(g["C"].shape[0], dtype=torch.long)
            g["num_tables"] = torch.tensor(1, dtype=torch.long)
        return g


class DatasetWithTag(Dataset):
    """Wraps a dataset and adds '_dataset' to each sample for multi-dataset eval routing."""

    def __init__(self, base_dataset, dataset_name):
        self.base = base_dataset
        self.dataset_name = dataset_name
        self.init_prompt = getattr(base_dataset, 'init_prompt', None)
        if hasattr(base_dataset, 'precomputed_split'):
            self.precomputed_split = base_dataset.precomputed_split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sample['_dataset'] = self.dataset_name
        return sample


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
            # mmap=False: graphs are now saved in legacy (non-zip) format (the zip
            # reader crashes under concurrent workers); legacy files can't mmap.
            sample['graph'] = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
        except (RuntimeError, EOFError) as e:
            raise RuntimeError(
                f"Corrupted precomputed graph file (delete and re-run precompute): {path}"
            ) from e
        return sample
