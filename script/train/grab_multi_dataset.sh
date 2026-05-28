#!/bin/bash
set -e

source .venv/bin/activate

PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .env

DATASETS=("multihiertt" "scitat" "mmqa" "tqa_bench" "atis" "geoquery" "spider_qa")

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Training grab_multi_table on: ${DATASET}"
    echo "=========================================="

    OUTPUT_DIR="${CKPT_DIR}/train/grab_multi/${DATASET}"
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "logs/train/grab_multi/${DATASET}"

    torchrun --nproc_per_node=2 --master_port 29571 src/train_parallel.py \
        --model_name "grab_multi_table" \
        --dataset "${DATASET}" \
        --multi_table "True" \
        --prompt_type "qwen" \
        --seed 42 \
        --lr 1e-4 \
        --wd 0.05 \
        --patience 3 \
        --batch_size 2 \
        --grad_steps 8 \
        --num_epochs 10 \
        --eval_batch_size 4 \
        --num_workers 8 \
        --llm_model_name "qwen3_4b" \
        --llm_frozen "True" \
        --llm_lora "False" \
        --max_txt_len 8192 \
        --max_new_tokens 128 \
        --projector_type "linear" \
        --table_encoder_frozen "False" \
        --gnn_base_model "Qwen/Qwen3-Embedding-0.6B" \
        --num_gnn_layers 1 \
        --num_row_latents 32 \
        --num_col_latents 32 \
        --num_val_latents 32 \
        --num_resampler_heads 4 \
        --num_resampler_layers 2 \
        --gnn_dropout 0.1 \
        --max_columns 512 \
        --max_hash_groups 16384 \
        --output_dir "${OUTPUT_DIR}" \
        --skip_list "src/skip_list.json"

    echo "Done: ${DATASET}"
done
