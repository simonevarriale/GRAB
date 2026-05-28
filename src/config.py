import argparse

from src.global_path import precomputed_graphs_root


def parse_args_table_llama():
    parser = argparse.ArgumentParser(description="GRAB")

    # ---- Experiment ---------------------------------------------------------
    exp = parser.add_argument_group("Experiment")
    exp.add_argument("--model_name", type=str, default='table_graph_llm')
    exp.add_argument("--project", type=str, default="project_table_graph_llm")
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument("--output_dir", type=str, default='output')

    # ---- Dataset ------------------------------------------------------------
    data = parser.add_argument_group("Dataset")
    data.add_argument("--dataset", type=str, default='wtq')
    data.add_argument("--test_dataset", type=str, default='')
    data.add_argument("--second_dataset", type=str, default='')
    data.add_argument("--prompt_type", type=str, default='llama2')
    data.add_argument("--num_workers", type=int, default=8)
    data.add_argument("--multi_table", type=str, default='True',
                      help="Multi-table mode: 'False' = single-table only, "
                           "'True' = all examples (single + multi), "
                           "'only' = multi-table examples only")
    data.add_argument("--max_tables", type=int, default=8,
                      help='Max number of tables per sample')
    data.add_argument("--filter_table_tokens", type=str, default='False',
                      help='If True, filter training samples whose linearized table exceeds max_txt_len tokens')
    data.add_argument("--max_rows_per_table", type=int, default=None,
                      help='Drop samples where any table exceeds this row count')
    data.add_argument("--dataset_max_rows", type=int, default=None,
                      help='Truncate each table to this many rows for model input')
    data.add_argument("--skip_list", type=str, default='',
                      help='Path to skip_list.json. Samples listed there are excluded from all splits.')
    data.add_argument("--precomputed_graphs", type=str, default='',
                      help='Precomputed graph folder name (e.g. "wtq_single"). '
                           'If relative, resolved as $PRECOMPUTED_GRAPHS/<dataset>/<name>. '
                           'Pass an absolute path to override the .env root entirely.')

    # ---- Training -----------------------------------------------------------
    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--grad_steps", type=int, default=2)
    train.add_argument("--num_epochs", type=int, default=10)
    train.add_argument("--warmup_epochs", type=float, default=1)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--wd", type=float, default=0.05)
    train.add_argument("--patience", type=float, default=2)

    # ---- Inference ----------------------------------------------------------
    inf = parser.add_argument_group("Inference")
    inf.add_argument("--eval_batch_size", type=int, default=8)
    inf.add_argument("--do_eval", type=str, default='False')

    # ---- LLM ----------------------------------------------------------------
    llm = parser.add_argument_group("LLM")
    llm.add_argument("--llm_model_name", type=str, default='7b')
    llm.add_argument("--llm_model_path", type=str, default='')
    llm.add_argument("--llm_frozen", type=str, default='True')
    llm.add_argument("--max_txt_len", type=int, default=512)
    llm.add_argument("--max_new_tokens", type=int, default=32)
    llm.add_argument("--llm_lora", type=str, default='True')
    llm.add_argument("--llm_ckpt_path", type=str, default='')
    llm.add_argument("--num_token", type=int, default=1)
    llm.add_argument("--load_in_4bit", type=str, default='False',
                     help='Load the LLM weights in nf4 4-bit (bitsandbytes). Frozen backbone only.')
    llm.add_argument("--enable_thinking", type=str, default='False',
                     help='Enable chain-of-thought reasoning for instruct models (e.g. Qwen3). '
                          'Predictions are stripped of <think>...</think> blocks before evaluation.')

    # ---- Projector ----------------------------------------------------------
    proj = parser.add_argument_group("Projector")
    proj.add_argument("--projector_type", type=str, default='linear',
                      choices=['linear', 'mlp', 'deep_mlp'],
                      help='Projector complexity: linear (1 layer), mlp (2 layers + GELU), '
                           'deep_mlp (3 layers + GELU + LayerNorm)')

    # ---- GNN encoder --------------------------------------------------------
    gnn = parser.add_argument_group("GNN encoder")
    gnn.add_argument("--gnn_base_model", type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                     help='Base transformer model for GNN encoder')
    gnn.add_argument("--num_gnn_layers", type=int, default=2,
                     help='Number of message-passing layers in GNN encoder')
    gnn.add_argument("--num_latents", type=int, default=64,
                     help='Number of latent vectors in K-token resampler')
    gnn.add_argument("--num_resampler_heads", type=int, default=8,
                     help='Number of attention heads in K-token resampler')
    gnn.add_argument("--num_resampler_layers", type=int, default=1,
                     help='Number of stacked cross-attention layers per latent group in K-token resampler')
    gnn.add_argument("--num_row_latents", type=int, default=4,
                     help='Number of latents for row cross-attention (question_split variant)')
    gnn.add_argument("--num_col_latents", type=int, default=2,
                     help='Number of latents for column cross-attention (question_split variant)')
    gnn.add_argument("--num_val_latents", type=int, default=2,
                     help='Number of latents for value cross-attention (question_split variant)')
    gnn.add_argument("--gnn_dropout", type=float, default=0.1,
                     help='Dropout rate in GNN message-passing layers')
    gnn.add_argument("--max_columns", type=int, default=64,
                     help='Max number of columns for learnable column-type embeddings')
    gnn.add_argument("--max_hash_groups", type=int, default=4096,
                     help='Size of pre-allocated hash buffer for value node initialization')
    gnn.add_argument("--max_header_len", type=int, default=32,
                     help='Max tokens per column header')
    gnn.add_argument("--row_max_len", type=int, default=128,
                     help='Max tokens per row string (row encoder)')
    gnn.add_argument("--gnn_hidden_size", type=int, default=384,
                     help='Hidden size for GNN encoder when no base model is used')
    gnn.add_argument("--question_max_len", type=int, default=512,
                     help='Max tokens for question encoding (precomputed row variant)')
    gnn.add_argument("--num_buckets", type=int, default=10,
                     help='Number of quantile bins for numeric value grouping in the table tokenizer')

    # ---- Table encoder ------------------------------------------------------
    tenc = parser.add_argument_group("Table encoder")
    tenc.add_argument("--table_encoder_frozen", type=str, default='False')
    tenc.add_argument("--table_encoder_name", type=str, default=None)

    # ---- Ablations ----------------------------------------------------------
    abl = parser.add_argument_group("Ablations")
    abl.add_argument("--no_question_conditioning", type=str, default='False',
                     help='Ablation: pass q_tokens=None to the resampler so latents attend '
                          'only to R/C/V table node features, not to question tokens.')
    abl.add_argument("--no_table_in_prompt", type=str, default='False',
                     help='Ablation: omit the linearized table from the LLM prompt so the model '
                          'must rely solely on GNN latents to access table content.')

    args = parser.parse_args()
    return args
