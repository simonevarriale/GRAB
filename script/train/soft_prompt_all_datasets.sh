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

MASTER_PORT=29572
NUM_LATENTS=10

# dataset -> multi_table flag
declare -A MULTI_TABLE
MULTI_TABLE["structProbe"]="False"
MULTI_TABLE["hitab"]="False"
MULTI_TABLE["wtq"]="False"
MULTI_TABLE["wikisql"]="False"
MULTI_TABLE["hctqa"]="False"
MULTI_TABLE["tabmwp"]="False"
MULTI_TABLE["multihiertt"]="True"
MULTI_TABLE["scitat"]="True"
MULTI_TABLE["mmqa"]="True"
MULTI_TABLE["tqa_bench"]="True"
MULTI_TABLE["atis"]="True"
MULTI_TABLE["geoquery"]="True"
MULTI_TABLE["spider_qa"]="True"

DATASETS=(
    "structProbe"
    "hitab"
    "wtq"
    "wikisql"
    "hctqa"
    "tabmwp"
    "multihiertt"
    "scitat"
    "mmqa"
    "tqa_bench"
    "atis"
    "geoquery"
    "spider_qa"
)

for DATASET in "${DATASETS[@]}"; do
    MT="${MULTI_TABLE[$DATASET]}"

    echo "=========================================="
    echo "Training soft_prompt_llm on: ${DATASET} (multi_table=${MT})"
    echo "=========================================="

    OUTPUT_DIR="${CKPT_DIR}/train/soft_prompt/${DATASET}"
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "logs/train/soft_prompt/${DATASET}"

    torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} src/train_parallel.py \
        --model_name "soft_prompt_llm" \
        --dataset "${DATASET}" \
        --multi_table "${MT}" \
        --prompt_type "qwen" \
        --seed 42 \
        --lr 1e-4 \
        --wd 0.05 \
        --patience 3 \
        --batch_size 2 \
        --grad_steps 8 \
        --num_epochs 10 \
        --eval_batch_size 4 \
        --llm_model_name "qwen3_4b" \
        --llm_frozen "True" \
        --llm_lora "False" \
        --max_txt_len 8192 \
        --max_new_tokens 64 \
        --projector_type "linear" \
        --table_encoder_name "" \
        --num_latents ${NUM_LATENTS} \
        --num_resampler_heads 8 \
        --gnn_hidden_size 1024 \
        --enable_thinking "False" \
        --skip_list "src/skip_list.json" \
        --output_dir "${OUTPUT_DIR}"

    echo "Done: ${DATASET}"
done
