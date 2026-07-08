"""Shared GNN architecture for GrabSingleTable and GrabMultiTable."""

from dataclasses import dataclass, asdict
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class GNNEncoderConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_gnn_layers: int = 2
    num_latents: int = 64
    num_resampler_heads: int = 8
    gnn_dropout: float = 0.1
    freeze_base_model: bool = False
    num_buckets: int = 10
    max_columns: int = 64
    max_hash_groups: int = 4096
    max_header_len: int = 32
    use_value_stats: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GNNEncoderSplitConfig(GNNEncoderConfig):
    num_row_latents: int = 4
    num_col_latents: int = 2
    num_val_latents: int = 2
    num_resampler_layers: int = 1

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GNNMultiTableEncoderSplitConfig(GNNEncoderSplitConfig):
    max_tables: int = 8

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# GNN layer
# ---------------------------------------------------------------------------

class TripartiteMessagePassing(nn.Module):
    """Three-node-type GNN: R ↔ V ↔ C (rows and columns linked through value nodes)."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        def _make_update():
            return nn.ModuleDict({
                "norm_agg": nn.LayerNorm(hidden_size),
                "norm_ffn": nn.LayerNorm(hidden_size),
                "mlp": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 2, hidden_size),
                ),
            })

        self.update_R = _make_update()
        self.update_C = _make_update()
        self.update_V = _make_update()
        self.drop = nn.Dropout(dropout)

    def forward(self, R, C, V, adj_rv, adj_cv):
        adj_vr = adj_rv.transpose(1, 2)
        adj_vc = adj_cv.transpose(1, 2)

        msg_V = torch.bmm(adj_vr, R) + torch.bmm(adj_vc, C)
        msg_R = torch.bmm(adj_rv, V)
        msg_C = torch.bmm(adj_cv, V)

        V = V + self.drop(self.update_V["norm_agg"](msg_V))
        V = V + self.drop(self.update_V["mlp"](self.update_V["norm_ffn"](V)))

        R = R + self.drop(self.update_R["norm_agg"](msg_R))
        R = R + self.drop(self.update_R["mlp"](self.update_R["norm_ffn"](R)))

        C = C + self.drop(self.update_C["norm_agg"](msg_C))
        C = C + self.drop(self.update_C["mlp"](self.update_C["norm_ffn"](C)))

        return R, C, V


# ---------------------------------------------------------------------------
# Split resampler
# ---------------------------------------------------------------------------

class _CrossAttnLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.norm_q   = nn.LayerNorm(hidden_size)
        self.norm_kv  = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, q, kv, kv_padding_mask):
        attn_out, _ = self.cross_attn(
            query=self.norm_q(q),
            key=self.norm_kv(kv),
            value=self.norm_kv(kv),
            key_padding_mask=kv_padding_mask,
        )
        out = q + attn_out
        out = out + self.ffn(self.norm_ffn(out))
        return out


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_latents, num_heads=8, dropout=0.1, num_layers=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size))
        self.layers = nn.ModuleList([
            _CrossAttnLayer(hidden_size, num_heads, dropout)
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, kv, kv_padding_mask):
        B = kv.size(0)
        out = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            out = layer(out, kv, kv_padding_mask)
        return out


class KTokenResamplerSplit(nn.Module):
    """Split resampler: R, C, V latent groups each question-conditioned."""

    def __init__(self, hidden_size, num_row_latents=4, num_col_latents=2,
                 num_val_latents=2, num_heads=8, dropout=0.1, num_layers=1):
        super().__init__()
        self.row_block = _CrossAttentionBlock(hidden_size, num_row_latents, num_heads, dropout, num_layers)
        self.col_block = _CrossAttentionBlock(hidden_size, num_col_latents, num_heads, dropout, num_layers)
        self.val_block = _CrossAttentionBlock(hidden_size, num_val_latents, num_heads, dropout, num_layers)

    def forward(self, R, row_mask, C, col_mask, V, val_mask,
                q_tokens=None, q_mask=None):
        has_q = q_tokens is not None and q_mask is not None

        if has_q:
            row_kv = torch.cat([R, q_tokens], dim=1)
            row_mask_inv = ~torch.cat([row_mask, q_mask], dim=1)
        else:
            row_kv = R
            row_mask_inv = ~row_mask
        row_out = self.row_block(row_kv, row_mask_inv)

        if has_q:
            col_kv = torch.cat([C, q_tokens], dim=1)
            col_mask_inv = ~torch.cat([col_mask, q_mask], dim=1)
        else:
            col_kv = C
            col_mask_inv = ~col_mask
        col_out = self.col_block(col_kv, col_mask_inv)

        if has_q:
            val_kv = torch.cat([V, q_tokens], dim=1)
            val_mask_inv = ~torch.cat([val_mask, q_mask], dim=1)
        else:
            val_kv = V
            val_mask_inv = ~val_mask
        val_out = self.val_block(val_kv, val_mask_inv)

        return torch.cat([row_out, col_out, val_out], dim=1)


# ---------------------------------------------------------------------------
# Precomputed encoders (no base transformer — R, C come precomputed)
# ---------------------------------------------------------------------------

class GNNTableEncoderPrecomputed(nn.Module):
    """Single-table GNN encoder that consumes precomputed R, C, q_tokens."""

    def __init__(self, config: GNNEncoderSplitConfig, hidden_size: int):
        super().__init__()
        if not isinstance(config, GNNEncoderSplitConfig):
            raise TypeError(f"Expected GNNEncoderSplitConfig, got {type(config)}")
        self.config = config
        self.hidden_size = int(hidden_size)

        self.message_passing_layers = nn.ModuleList([
            TripartiteMessagePassing(self.hidden_size, dropout=self.config.gnn_dropout)
            for _ in range(self.config.num_gnn_layers)
        ])
        self.resampler = KTokenResamplerSplit(
            self.hidden_size,
            num_row_latents=self.config.num_row_latents,
            num_col_latents=self.config.num_col_latents,
            num_val_latents=self.config.num_val_latents,
            num_heads=self.config.num_resampler_heads,
            num_layers=getattr(self.config, "num_resampler_layers", 1),
        )

        gen = torch.Generator()
        gen.manual_seed(42)
        hash_buffer = torch.randn(self.config.max_hash_groups, self.hidden_size, generator=gen)
        self.register_buffer("hash_buffer", hash_buffer)

        self.col_val_proj = nn.Linear(self.hidden_size, self.hidden_size)

        if self.config.use_value_stats:
            self.value_stats_proj = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

    def _init_value_nodes(self, group_to_col, max_groups, C, value_stats=None):
        B = group_to_col.shape[0]
        if max_groups <= self.hash_buffer.shape[0]:
            hash_vecs = self.hash_buffer[:max_groups].unsqueeze(0).expand(B, -1, -1)
        else:
            gen = torch.Generator(device=group_to_col.device)
            gen.manual_seed(42)
            hash_vecs = torch.randn(max_groups, self.hidden_size,
                                    generator=gen, device=group_to_col.device)
            hash_vecs = hash_vecs.unsqueeze(0).expand(B, -1, -1)
        # Learnable projection of content-based column embeddings: permutation invariant
        # (keyed by column header text, not index) and task-adaptable via col_val_proj.
        col_emb = C.gather(
            dim=1,
            index=group_to_col.unsqueeze(-1).expand(-1, -1, C.shape[-1])
        )
        V = hash_vecs + self.col_val_proj(col_emb)
        if self.config.use_value_stats and value_stats is not None:
            V = V + self.value_stats_proj(value_stats)
        return V

    def forward(self, *, R, row_mask, C, col_mask, adj, adj_cv, group_to_col,
                value_stats=None, q_tokens=None, q_mask=None):
        max_groups = adj.shape[2]
        V = self._init_value_nodes(group_to_col, max_groups, C, value_stats)

        B, max_rows, _ = R.shape
        num_cols = C.shape[1]

        def _pad(mat, target_rows, target_cols):
            if mat.shape[1] == target_rows and mat.shape[2] == target_cols:
                return mat
            out = torch.zeros(B, target_rows, target_cols, device=mat.device, dtype=mat.dtype)
            out[:, :mat.shape[1], :mat.shape[2]] = mat
            return out

        adj    = _pad(adj,    max_rows, max_groups)
        adj_cv = _pad(adj_cv, num_cols, max_groups)

        for layer in self.message_passing_layers:
            R, C, V = layer(R, C, V, adj, adj_cv)

        val_mask = adj.sum(dim=1) > 0
        return self.resampler(R, row_mask, C, col_mask, V, val_mask,
                              q_tokens=q_tokens, q_mask=q_mask)


class GNNMultiTableEncoderWithQuestionSplitPrecomputed(GNNTableEncoderPrecomputed):
    """Multi-table precomputed encoder. Adds a learnable table embedding to R and C."""

    def __init__(self, config: GNNMultiTableEncoderSplitConfig, hidden_size: int):
        super().__init__(config=config, hidden_size=hidden_size)
        max_tables = getattr(config, 'max_tables', 8)
        self.table_embed = nn.Embedding(max_tables, self.hidden_size)

    def forward(self, *, R, row_mask, C, col_mask, adj, adj_cv, group_to_col,
                value_stats=None, q_tokens=None, q_mask=None,
                row_table_ids=None, col_table_ids=None, num_tables=None):
        if row_table_ids is not None:
            R = R + self.table_embed(row_table_ids)
        if col_table_ids is not None:
            C = C + self.table_embed(col_table_ids)
        return super().forward(
            R=R, row_mask=row_mask, C=C, col_mask=col_mask,
            adj=adj, adj_cv=adj_cv, group_to_col=group_to_col,
            value_stats=value_stats, q_tokens=q_tokens, q_mask=q_mask,
        )
