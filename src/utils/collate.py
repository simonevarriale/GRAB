import torch


def collate_graph_batch(graphs):
    """Pad and stack a list of per-sample graph dicts into one batched dict.

    Runs inside DataLoader workers so padding is off the training-thread critical
    path. Converts float16 stored tensors to float32 here to avoid casting in
    the model forward.
    """
    B = len(graphs)
    H = graphs[0]["R"].shape[1]

    max_r  = max(g["R"].shape[0]       for g in graphs)
    max_c  = max(g["C"].shape[0]       for g in graphs)
    max_ng = max(g["adj"].shape[1]     for g in graphs)

    has_q        = "q_tokens"      in graphs[0]
    has_vs       = "value_stats"   in graphs[0]
    has_tids     = "row_table_ids" in graphs[0]

    max_lq = max(g["q_tokens"].shape[0] for g in graphs) if has_q else 0

    R        = torch.zeros(B, max_r,  H,     dtype=torch.float32)
    C        = torch.zeros(B, max_c,  H,     dtype=torch.float32)
    row_mask = torch.zeros(B, max_r,         dtype=torch.bool)
    col_mask = torch.zeros(B, max_c,         dtype=torch.bool)
    adj      = torch.zeros(B, max_r,  max_ng, dtype=torch.float32)
    adj_cv   = torch.zeros(B, max_c,  max_ng, dtype=torch.float32)
    g2c      = torch.zeros(B, max_ng,         dtype=torch.long)

    q_tokens    = torch.zeros(B, max_lq, H,     dtype=torch.float32) if has_q   else None
    q_mask      = torch.zeros(B, max_lq,         dtype=torch.bool)   if has_q   else None
    value_stats = torch.zeros(B, max_ng, 1,      dtype=torch.float32) if has_vs  else None
    row_tids    = torch.zeros(B, max_r,           dtype=torch.long)   if has_tids else None
    col_tids    = torch.zeros(B, max_c,           dtype=torch.long)   if has_tids else None
    num_tables  = torch.zeros(B,                  dtype=torch.long)   if has_tids else None

    for i, g in enumerate(graphs):
        r  = g["R"].shape[0]
        c  = g["C"].shape[0]
        ng = g["adj"].shape[1]

        R[i, :r]            = g["R"].float()
        row_mask[i, :r]     = g["row_mask"]
        C[i, :c]            = g["C"].float()
        col_mask[i, :c]     = g["col_mask"]
        adj[i, :r, :ng]     = g["adj"]
        adj_cv[i, :c, :ng]  = g["adj_cv"]
        g2c[i, :ng]         = g["group_to_col"]

        if has_q:
            lq = g["q_tokens"].shape[0]
            q_tokens[i, :lq] = g["q_tokens"].float()
            q_mask[i, :lq]   = g["q_mask"]

        if has_vs:
            value_stats[i, :ng] = g["value_stats"]

        if has_tids:
            row_tids[i, :r] = g["row_table_ids"]
            col_tids[i, :c] = g["col_table_ids"]
            num_tables[i]   = g["num_tables"]

    out = dict(R=R, row_mask=row_mask, C=C, col_mask=col_mask,
               adj=adj, adj_cv=adj_cv, group_to_col=g2c)
    if has_q:
        out["q_tokens"] = q_tokens
        out["q_mask"]   = q_mask
    if has_vs:
        out["value_stats"] = value_stats
    if has_tids:
        out["row_table_ids"] = row_tids
        out["col_table_ids"] = col_tids
        out["num_tables"]    = num_tables
    return out


def collate_fn(batch):
    """Transpose a list of sample dicts into a dict of lists.

    Graph fields (key 'graph') are padded and stacked into batched tensors by
    collate_graph_batch so that encoding can start immediately on the GPU without
    any CPU padding work on the training thread.
    """
    if not batch:
        return {}
    result = {key: [d[key] for d in batch] for key in batch[0]}
    if "graph" in result and isinstance(result["graph"][0], dict):
        result["graph"] = collate_graph_batch(result["graph"])
    return result
