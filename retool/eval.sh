#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

MODEL_DIR=/data/agenthle/baohao/agentic_opd/retool/model
DATA_DIR=/data/agenthle/baohao/agentic_opd/retool/data

MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/qwen3-4b-sft-rl}
TP_SIZE=${TP_SIZE:-2}
NUM_SAMPLES=${NUM_SAMPLES:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
PORT=${PORT:-30000}
OUTPUT=${OUTPUT:-${MODEL_PATH}/eval_aime2024.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-${MODEL_PATH}/eval_aime2024_summary.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-"subprocess"}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-"32"}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-"300"}
export LOGNAME=${LOGNAME:-"user"}

cd "${SCRIPT_DIR}"

python eval.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATA_DIR}/aime-2024/aime-2024.jsonl" \
    --tp-size ${TP_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature 1.0 \
    --top-p 1.0 \
    --port ${PORT} \
    --max-concurrent 16 \
    --output "${OUTPUT}" \
    --summary-output "${SUMMARY_OUTPUT}" \
    --print-turns
