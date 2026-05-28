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

DATA_ROOT="${PRECOMPUTED_GRAPHS}"
GNN_MODEL="Qwen/Qwen3-Embedding-0.6B"
SKIP_LIST="src/skip_list.json"

QUESTION_MAX_LEN=256

declare -A ROW_MAX=(
    [structProbe]=1024
    [hitab]=2304
    [wtq]=1024
    [wikisql]=512
    [hctqa]=1024
    [tabmwp]=128
)
declare -A HDR_MAX=(
    [structProbe]=32
    [hitab]=256
    [wtq]=64
    [wikisql]=256
    [hctqa]=64
    [tabmwp]=16
)

TORCHRUN="torchrun --nproc_per_node=2 --master_port 29600"

run_single() {
    local DATASET="$1"
    local ROW_MAX="${ROW_MAX[$DATASET]}"
    local HDR_MAX="${HDR_MAX[$DATASET]}"
    local OUT_DIR="${DATA_ROOT}/${DATASET}/precomputed_graphs_row_qwen_final"
    echo "=== [single] ${DATASET}  row_max=${ROW_MAX}  hdr_max=${HDR_MAX}  → ${OUT_DIR} ==="

    ${TORCHRUN} src/precompute_single_table.py \
        --dataset            "${DATASET}" \
        --prompt_type        "qwen" \
        --gnn_base_model     "${GNN_MODEL}" \
        --row_max_len        "${ROW_MAX}" \
        --max_header_len     "${HDR_MAX}" \
        --question_max_len   "${QUESTION_MAX_LEN}" \
        --row_batch_size     32 \
        --sample_batch_size  32 \
        --num_buckets        10 \
        --precomputed_graphs "${OUT_DIR}" \
        --skip_list          "${SKIP_LIST}"
}

DATASETS=("structProbe" "hitab" "wtq" "wikisql" "hctqa" "tabmwp")

mkdir -p logs/precompute

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Precomputing single-table graphs: ${DATASET}"
    echo "=========================================="

    mkdir -p "logs/precompute/${DATASET}"
    run_single "${DATASET}"

    echo "Done: ${DATASET}"
done
