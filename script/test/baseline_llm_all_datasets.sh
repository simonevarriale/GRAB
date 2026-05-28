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

MASTER_PORT=29573

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
    echo "Evaluating base_llm on: ${DATASET} (multi_table=${MT})"
    echo "=========================================="

    OUTPUT_DIR="${CKPT_DIR}/test/base_llm/${DATASET}"
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "logs/test/base_llm/${DATASET}"

    torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} src/test.py \
        --model_name "base_llm" \
        --dataset "${DATASET}" \
        --multi_table "${MT}" \
        --prompt_type "qwen" \
        --seed 42 \
        --eval_batch_size 4 \
        --num_workers 8 \
        --llm_model_name "qwen3_4b" \
        --llm_frozen "True" \
        --llm_lora "False" \
        --max_txt_len 8192 \
        --max_new_tokens 128 \
        --enable_thinking "False" \
        --table_encoder_name "None" \
        --output_dir "${OUTPUT_DIR}" \
        --skip_list "src/skip_list.json"

    echo "Done: ${DATASET}"
done
