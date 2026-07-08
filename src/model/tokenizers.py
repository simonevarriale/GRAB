"""Table tokenizers for GNN encoder variants."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.load_local_model import load_tokenizer_local_or_hf


# ---------------------------------------------------------------------------
# Base col-based tokenizer
# ---------------------------------------------------------------------------

class TableTokenizerBase:
    """Col-based table tokenizer: cell tokens tagged with row/col/group IDs."""

    def __init__(self, base_model_name, max_length=None, num_buckets=10,
                 max_header_len=32):
        self.tokenizer = load_tokenizer_local_or_hf(base_model_name)
        self.num_buckets = num_buckets
        self.max_header_len = max_header_len

        if max_length is None or max_length <= 0 or max_length > self.tokenizer.model_max_length:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length

    def _tokenize_headers(self, columns):
        """Tokenize column names → padded tensors [num_cols, max_header_len]."""
        enc = self.tokenizer(
            list(columns),
            padding="max_length",
            truncation=True,
            max_length=self.max_header_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def encode_table(self, df: pd.DataFrame, include_column_names=True):
        num_rows, num_cols = df.shape

        if include_column_names:
            text_matrix = df.astype(str).apply(
                lambda col: col.name + ": " + col, axis=0
            ).to_numpy()
        else:
            text_matrix = df.astype(str).to_numpy()

        flat_texts = text_matrix.flatten(order='C').tolist()
        batch_enc = self.tokenizer(flat_texts, add_special_tokens=False, verbose=False)
        all_cell_tokens = batch_enc['input_ids']

        cell_lengths = np.array([len(t) for t in all_cell_tokens])
        cell_lengths_matrix = cell_lengths.reshape(num_rows, num_cols)
        cumulative_len = np.cumsum(cell_lengths_matrix.sum(axis=1))

        valid_rows_count = np.searchsorted(cumulative_len, self.max_length, side='right')
        force_truncate = False
        if valid_rows_count == 0 and num_rows > 0:
            valid_rows_count = 1
            force_truncate = True

        cells_to_keep = valid_rows_count * num_cols
        final_cell_tokens = all_cell_tokens[:cells_to_keep]

        group_ids_matrix = np.zeros((valid_rows_count, num_cols), dtype=int)
        val_to_id, group_to_col_map = {}, {}
        current_gid = 0

        bin_edges, numeric_cache = {}, {}
        for c in range(num_cols):
            numeric_vals = pd.to_numeric(df.iloc[:, c], errors='coerce')
            if numeric_vals.notna().mean() > 0.8:
                numeric_cache[c] = numeric_vals
                clean = numeric_vals.dropna().values
                n_unique = len(np.unique(clean))
                if n_unique > self.num_buckets:
                    n_bins = max(2, int(np.ceil(np.sqrt(n_unique * self.num_buckets))))
                    bin_edges[c] = np.quantile(
                        clean, np.linspace(0, 1, n_bins + 1)[1:-1]
                    )

        for c in range(num_cols):
            is_numeric = c in bin_edges
            num_vals = numeric_cache.get(c)
            for r in range(valid_rows_count):
                if is_numeric and pd.notna(num_vals.iloc[r]):
                    bin_idx = int(np.searchsorted(bin_edges[c], num_vals.iloc[r]))
                    key = f"col_{c}_bin_{bin_idx}"
                else:
                    key = f"col_{c}_{str(df.iloc[r, c])}"
                if key not in val_to_id:
                    val_to_id[key] = current_gid
                    group_to_col_map[current_gid] = c
                    current_gid += 1
                group_ids_matrix[r, c] = val_to_id[key]

        group_ids_flat = group_ids_matrix.flatten(order='C')
        max_groups = int(current_gid)

        adj_rv = np.zeros((valid_rows_count, max_groups), dtype=np.float32)
        for r in range(valid_rows_count):
            for c in range(num_cols):
                adj_rv[r, group_ids_matrix[r, c]] = 1.0

        adj_cv = np.zeros((num_cols, max_groups), dtype=np.float32)
        for gid, col_idx in group_to_col_map.items():
            adj_cv[col_idx, gid] = 1.0

        row_indices_map = np.repeat(np.arange(valid_rows_count), num_cols)
        col_indices_map = np.tile(np.arange(num_cols), valid_rows_count)
        valid_lengths = cell_lengths[:cells_to_keep]

        row_ids = np.repeat(row_indices_map, valid_lengths)
        col_ids = np.repeat(col_indices_map, valid_lengths)
        group_ids = np.repeat(group_ids_flat[:cells_to_keep], valid_lengths)
        input_ids = np.concatenate(final_cell_tokens)

        if force_truncate and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            row_ids = row_ids[:self.max_length]
            col_ids = col_ids[:self.max_length]
            group_ids = group_ids[:self.max_length]

        group_to_col_arr = np.zeros(max_groups, dtype=int)
        for gid, col_idx in group_to_col_map.items():
            group_to_col_arr[gid] = col_idx

        value_stats = adj_rv.sum(axis=0, keepdims=True).T  # [max_groups, 1]

        col_header_ids, col_header_mask = self._tokenize_headers(df.columns)

        return {
            "input_ids":       torch.as_tensor(input_ids, dtype=torch.long).unsqueeze(0),
            "attention_mask":  torch.ones(1, len(input_ids), dtype=torch.long),
            "row_ids":         torch.as_tensor(row_ids,   dtype=torch.long).unsqueeze(0),
            "col_ids":         torch.as_tensor(col_ids,   dtype=torch.long).unsqueeze(0),
            "group_ids":       torch.as_tensor(group_ids, dtype=torch.long).unsqueeze(0),
            "adj":             torch.as_tensor(adj_rv, dtype=torch.float32).unsqueeze(0),
            "adj_cv":          torch.as_tensor(adj_cv, dtype=torch.float32).unsqueeze(0),
            "group_to_col":    torch.as_tensor(group_to_col_arr, dtype=torch.long).unsqueeze(0),
            "value_stats":     torch.as_tensor(value_stats, dtype=torch.float32).unsqueeze(0),
            "col_header_ids":  col_header_ids.unsqueeze(0),
            "col_header_mask": col_header_mask.unsqueeze(0),
        }


class TableTokenizerWithQuestion(TableTokenizerBase):
    """Extends TableTokenizerBase with question encoding."""

    def encode_question(self, question: str):
        enc = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "question_ids":  enc["input_ids"],      # [1, L]
            "question_mask": enc["attention_mask"],  # [1, L]
        }


# ---------------------------------------------------------------------------
# Row-based tokenizer (used by GrabSingleTable precompute)
# ---------------------------------------------------------------------------

class TableTokenizerRow:
    """Row-based tokenizer: each row becomes one sequence for independent encoding."""

    def __init__(self, base_model_name, max_length=None, num_buckets=10,
                 max_header_len=32, row_max_len=128):
        self.tokenizer = load_tokenizer_local_or_hf(base_model_name)
        self.num_buckets = num_buckets
        self.max_header_len = max_header_len
        self.row_max_len = row_max_len

        if max_length is None or max_length <= 0 or max_length > self.tokenizer.model_max_length:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length

    def _tokenize_headers(self, columns):
        enc = self.tokenizer(
            list(columns),
            padding="max_length",
            truncation=True,
            max_length=self.max_header_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def _build_row_strings(self, df: pd.DataFrame):
        cols = list(df.columns)
        if not cols:
            return [""] * len(df)
        combined = np.char.add(f"{cols[0]}: ", df.iloc[:, 0].astype(str).to_numpy())
        for c in range(1, len(cols)):
            combined = np.char.add(
                np.char.add(combined, f" {cols[c]}: "),
                df.iloc[:, c].astype(str).to_numpy(),
            )
        return combined.tolist()

    def encode_table(self, df: pd.DataFrame, include_column_names=True):
        num_rows, num_cols = df.shape

        row_texts = self._build_row_strings(df)
        row_enc = self.tokenizer(
            row_texts,
            padding="max_length",
            truncation=True,
            max_length=self.row_max_len,
            return_tensors="pt",
        )
        row_input_ids = row_enc["input_ids"]
        row_attention_mask = row_enc["attention_mask"]

        group_ids_matrix = np.empty((num_rows, num_cols), dtype=np.int64)
        group_to_col_list: list = []
        current_gid = 0

        for c in range(num_cols):
            numeric_vals = pd.to_numeric(df.iloc[:, c], errors='coerce')
            str_vals = df.iloc[:, c].astype(str).to_numpy()

            if numeric_vals.notna().mean() > 0.8:
                clean = numeric_vals.dropna().to_numpy()
                n_unique = len(np.unique(clean))
                if n_unique > self.num_buckets:
                    n_bins = max(2, int(np.ceil(np.sqrt(n_unique * self.num_buckets))))
                    edges = np.quantile(clean, np.linspace(0, 1, n_bins + 1)[1:-1])
                    nan_mask = numeric_vals.isna().to_numpy()
                    bin_idx = np.searchsorted(edges, numeric_vals.fillna(0.0).to_numpy())
                    keys = np.where(nan_mask, str_vals, bin_idx.astype(str))
                else:
                    keys = str_vals
            else:
                keys = str_vals

            local_ids, uniques = pd.factorize(keys, sort=False)
            n_uniq = len(uniques)
            group_ids_matrix[:, c] = local_ids + current_gid
            group_to_col_list.extend([c] * n_uniq)
            current_gid += n_uniq

        max_groups = current_gid

        adj_rv = np.zeros((num_rows, max_groups), dtype=np.float32)
        adj_rv[np.arange(num_rows)[:, None], group_ids_matrix] = 1.0

        adj_cv = np.zeros((num_cols, max_groups), dtype=np.float32)
        adj_cv[np.arange(num_cols)[:, None], group_ids_matrix.T] = 1.0

        group_to_col_arr = np.array(group_to_col_list, dtype=np.int64)

        value_stats = adj_rv.sum(axis=0, keepdims=True).T  # [max_groups, 1]

        col_header_ids, col_header_mask = self._tokenize_headers(df.columns)

        return {
            "row_input_ids":      row_input_ids.unsqueeze(0),
            "row_attention_mask": row_attention_mask.unsqueeze(0),
            "adj":                torch.as_tensor(adj_rv, dtype=torch.float32).unsqueeze(0),
            "adj_cv":             torch.as_tensor(adj_cv, dtype=torch.float32).unsqueeze(0),
            "group_to_col":       torch.as_tensor(group_to_col_arr, dtype=torch.long).unsqueeze(0),
            "value_stats":        torch.as_tensor(value_stats, dtype=torch.float32).unsqueeze(0),
            "col_header_ids":     col_header_ids.unsqueeze(0),
            "col_header_mask":    col_header_mask.unsqueeze(0),
        }

    def encode_question(self, question: str):
        enc = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "question_ids":  enc["input_ids"],
            "question_mask": enc["attention_mask"],
        }


# ---------------------------------------------------------------------------
# FK-aware multi-table tokenizer (used by GrabMultiTable precompute)
# ---------------------------------------------------------------------------

class MultiTableTokenizerSplitRow(TableTokenizerWithQuestion):
    """FK-aware multi-table tokenizer with per-row sequences.

    encode_tables() builds one unified graph across all tables, merging columns
    linked by foreign keys into shared column nodes.
    """

    def __init__(self, base_model_name, max_length=None, num_buckets=10,
                 max_header_len=32, max_row_len=128):
        super().__init__(base_model_name, max_length=max_length,
                         num_buckets=num_buckets, max_header_len=max_header_len)
        self.max_row_len = max_row_len

    @staticmethod
    def _stringify_tables(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        return [df.astype(str, copy=False) for df in dfs]

    def _encode_row_sequences(self, df: pd.DataFrame,
                               valid_rows_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if valid_rows_count == 0:
            return (
                torch.zeros((0, self.max_row_len), dtype=torch.long),
                torch.zeros((0, self.max_row_len), dtype=torch.long),
            )

        col_prefixes = [f"{str(col)}: " for col in df.columns]
        row_texts = [
            " ".join(prefix + value for prefix, value in zip(col_prefixes, row))
            for row in df.iloc[:valid_rows_count].itertuples(index=False, name=None)
        ]

        enc = self.tokenizer(
            row_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_row_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    @staticmethod
    def _resolve_table_names(
        dfs: List[pd.DataFrame],
        table_names: Optional[List[str]],
    ) -> List[str]:
        if table_names is None or len(table_names) != len(dfs):
            return [f"table_{i}" for i in range(len(dfs))]
        return [str(name) for name in table_names]

    @staticmethod
    def _build_shared_column_map(
        dfs: List[pd.DataFrame],
        table_names: List[str],
        foreign_keys: Optional[List[List[List[str]]]],
    ) -> Tuple[Dict[Tuple[int, int], int], List[str], List[int]]:
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def find(x):
            root = parent.setdefault(x, x)
            if root != x:
                parent[x] = find(root)
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ti, df in enumerate(dfs):
            for ci in range(df.shape[1]):
                parent[(ti, ci)] = (ti, ci)

        if foreign_keys:
            table_to_idx = {name: i for i, name in enumerate(table_names)}
            table_to_idx_lower = {name.lower(): i for i, name in enumerate(table_names)}
            col_name_maps = []
            col_name_maps_lower = []
            for df in dfs:
                names = [str(col) for col in df.columns]
                col_name_maps.append({name: i for i, name in enumerate(names)})
                col_name_maps_lower.append({name.lower(): i for i, name in enumerate(names)})

            for pair in foreign_keys:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                left, right = pair
                if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                    continue
                if len(left) != 2 or len(right) != 2:
                    continue

                t1_name, c1_name = str(left[0]), str(left[1])
                t2_name, c2_name = str(right[0]), str(right[1])

                t1 = table_to_idx.get(t1_name, table_to_idx_lower.get(t1_name.lower()))
                t2 = table_to_idx.get(t2_name, table_to_idx_lower.get(t2_name.lower()))
                if t1 is None or t2 is None:
                    continue

                c1 = col_name_maps[t1].get(c1_name, col_name_maps_lower[t1].get(c1_name.lower()))
                c2 = col_name_maps[t2].get(c2_name, col_name_maps_lower[t2].get(c2_name.lower()))
                if c1 is None or c2 is None:
                    continue

                union((t1, c1), (t2, c2))

        local_to_shared: Dict[Tuple[int, int], int] = {}
        members_by_shared: List[List[Tuple[int, int]]] = []

        for ti, df in enumerate(dfs):
            for ci in range(df.shape[1]):
                root = find((ti, ci))
                if root not in local_to_shared:
                    shared_idx = len(members_by_shared)
                    local_to_shared[root] = shared_idx
                    members_by_shared.append([])
                shared_idx = local_to_shared[root]
                local_to_shared[(ti, ci)] = shared_idx
                members_by_shared[shared_idx].append((ti, ci))

        header_texts: List[str] = []
        col_table_ids: List[int] = []
        for members in members_by_shared:
            first_ti, first_ci = members[0]
            first_name = str(dfs[first_ti].columns[first_ci])
            unique_names = {str(dfs[ti].columns[ci]) for ti, ci in members}
            if len(unique_names) == 1:
                header_text = first_name
            else:
                aliases = [f"{table_names[ti]}.{str(dfs[ti].columns[ci])}" for ti, ci in members]
                header_text = " | ".join(aliases)
            header_texts.append(header_text)
            col_table_ids.append(first_ti)

        return local_to_shared, header_texts, col_table_ids

    def encode_tables(self, dfs: List[pd.DataFrame],
                      question: Optional[str] = None,
                      foreign_keys: Optional[List[List[List[str]]]] = None,
                      table_names: Optional[List[str]] = None) -> dict:
        dfs = self._stringify_tables(dfs)
        resolved_table_names = self._resolve_table_names(dfs, table_names)
        local_to_shared, header_texts, shared_col_tids = self._build_shared_column_map(
            dfs, resolved_table_names, foreign_keys,
        )

        total_cols = len(header_texts)
        total_rows = sum(len(df) for df in dfs)

        current_group_offset = 0
        row_offset = 0

        row_table_ids_parts = []
        group_to_col_parts = []
        row_seq_ids, row_seq_masks = [], []
        table_infos = []

        for ti, df in enumerate(dfs):
            num_rows, num_cols = df.shape
            valid_rows_count = num_rows

            if num_rows == 0 or num_cols == 0:
                group_ids_matrix = np.zeros((0, num_cols), dtype=np.int64)
                local_num_groups = 0
                local_group_to_shared_col = np.zeros((0,), dtype=np.int64)
            else:
                group_ids_matrix = np.empty((valid_rows_count, num_cols), dtype=np.int64)
                shared_col_parts = []
                current_local_gid = 0

                for c in range(num_cols):
                    numeric_vals = pd.to_numeric(df.iloc[:, c], errors="coerce")
                    str_vals = df.iloc[:, c].astype(str).to_numpy()

                    if numeric_vals.notna().mean() > 0.8:
                        clean = numeric_vals.dropna().to_numpy()
                        n_unique = len(np.unique(clean))
                        if n_unique > self.num_buckets:
                            n_bins = max(2, int(np.ceil(np.sqrt(n_unique * self.num_buckets))))
                            edges = np.quantile(clean, np.linspace(0, 1, n_bins + 1)[1:-1])
                            nan_mask = numeric_vals.isna().to_numpy()
                            bin_idx = np.searchsorted(edges, numeric_vals.fillna(0.0).to_numpy())
                            keys = np.where(nan_mask, str_vals, bin_idx.astype(str))
                        else:
                            keys = str_vals
                    else:
                        keys = str_vals

                    local_ids, uniques = pd.factorize(keys, sort=False)
                    n_uniq = len(uniques)
                    group_ids_matrix[:, c] = local_ids + current_local_gid
                    shared_col_parts.append(
                        np.full(n_uniq, local_to_shared[(ti, c)], dtype=np.int64)
                    )
                    current_local_gid += n_uniq

                local_num_groups = current_local_gid
                local_group_to_shared_col = (
                    np.concatenate(shared_col_parts)
                    if shared_col_parts else np.zeros((0,), dtype=np.int64)
                )

            row_table_ids_parts.append(np.full(valid_rows_count, ti, dtype=np.int64))
            group_to_col_parts.append(local_group_to_shared_col)

            ids, masks = self._encode_row_sequences(df, valid_rows_count)
            row_seq_ids.append(ids)
            row_seq_masks.append(masks)

            table_infos.append({
                "row_offset": row_offset,
                "num_rows": valid_rows_count,
                "num_cols": num_cols,
                "group_offset": current_group_offset,
                "group_ids_matrix": group_ids_matrix,
            })

            row_offset += valid_rows_count
            current_group_offset += local_num_groups

        total_groups = current_group_offset

        input_ids = np.zeros((0,), dtype=np.int64)
        row_ids = np.zeros((0,), dtype=np.int64)
        col_ids = np.zeros((0,), dtype=np.int64)
        group_ids = np.zeros((0,), dtype=np.int64)
        row_tids = (
            np.concatenate(row_table_ids_parts)
            if row_table_ids_parts else np.zeros((0,), dtype=np.int64)
        )
        group_to_col = (
            np.concatenate(group_to_col_parts)
            if group_to_col_parts else np.zeros((0,), dtype=np.int64)
        )

        adj_rv = np.zeros((total_rows, total_groups), dtype=np.float32)
        adj_cv = np.zeros((total_cols, total_groups), dtype=np.float32)
        for ti, info in enumerate(table_infos):
            if info["num_rows"] == 0 or info["num_cols"] == 0:
                continue
            gids = info["group_ids_matrix"] + info["group_offset"]
            row_idx = np.arange(info["num_rows"], dtype=np.int64) + info["row_offset"]
            shared_col_idx = np.array(
                [local_to_shared[(ti, c)] for c in range(info["num_cols"])], dtype=np.int64
            )
            adj_rv[row_idx[:, None], gids] = 1.0
            adj_cv[shared_col_idx[None, :], gids] = 1.0

        value_stats = (
            adj_rv.sum(axis=0, keepdims=True).T
            if total_groups > 0
            else np.zeros((0, 1), dtype=np.float32)
        )

        if total_cols > 0:
            col_header_ids, col_header_mask = self._tokenize_headers(header_texts)
        else:
            col_header_ids = torch.zeros((0, self.max_header_len), dtype=torch.long)
            col_header_mask = torch.zeros((0, self.max_header_len), dtype=torch.long)

        result = {
            "input_ids":       torch.as_tensor(input_ids,    dtype=torch.long).unsqueeze(0),
            "attention_mask":  torch.ones(1, len(input_ids), dtype=torch.long),
            "row_ids":         torch.as_tensor(row_ids,      dtype=torch.long).unsqueeze(0),
            "col_ids":         torch.as_tensor(col_ids,      dtype=torch.long).unsqueeze(0),
            "group_ids":       torch.as_tensor(group_ids,    dtype=torch.long).unsqueeze(0),
            "adj":             torch.as_tensor(adj_rv,       dtype=torch.float32).unsqueeze(0),
            "adj_cv":          torch.as_tensor(adj_cv,       dtype=torch.float32).unsqueeze(0),
            "group_to_col":    torch.as_tensor(group_to_col, dtype=torch.long).unsqueeze(0),
            "value_stats":     torch.as_tensor(value_stats,  dtype=torch.float32).unsqueeze(0),
            "col_header_ids":  col_header_ids.unsqueeze(0),
            "col_header_mask": col_header_mask.unsqueeze(0),
            "row_table_ids":   torch.as_tensor(row_tids, dtype=torch.long).unsqueeze(0),
            "col_table_ids":   torch.as_tensor(np.array(shared_col_tids, dtype=np.int64), dtype=torch.long).unsqueeze(0),
            "num_tables":      torch.tensor(len(dfs), dtype=torch.long).unsqueeze(0),
            "row_input_ids":   torch.cat(row_seq_ids,   dim=0).unsqueeze(0),
            "row_attn_mask":   torch.cat(row_seq_masks, dim=0).unsqueeze(0),
        }

        if question is not None:
            result.update(self.encode_question(question))

        return result
