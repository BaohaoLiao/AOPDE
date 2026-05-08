#!/bin/bash
# Evaluate against an SGLang server that you started separately, e.g.:
#
#   MODEL_PATH=/path/to/hf TP_SIZE=2 PORT=30000 \
#       CUDA_VISIBLE_DEVICES=0,1 bash retool/sglang_serve.sh
#
# Then in another shell:
#
#   MODEL_PATH=/path/to/hf PORT=30000 bash retool/eval.sh
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

MODEL_DIR=/data/agenthle/baohao/agentic_opd/retool/model
DATA_DIR=/data/agenthle/baohao/agentic_opd/retool/data

MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/qwen3-4b-sft-rl}
NUM_SAMPLES=${NUM_SAMPLES:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
OUTPUT=${OUTPUT:-${MODEL_PATH}/eval_aime2024.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-${MODEL_PATH}/eval_aime2024_summary.json}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-1800}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-60}

export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-"subprocess"}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-"32"}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-"60"}
export USER=${USER:-"user"}
export LOGNAME=${LOGNAME:-"$USER"}
export HOME=${HOME:-/tmp}

cd "${SCRIPT_DIR}"

python eval.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATA_DIR}/aime-2024/aime-2024.jsonl" \
    --num-samples ${NUM_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature 1.0 \
    --top-p 1.0 \
    --host "${HOST}" \
    --port ${PORT} \
    --server-wait-timeout ${SERVER_WAIT_TIMEOUT} \
    --max-concurrent 16 \
    --output "${OUTPUT}" \
    --summary-output "${SUMMARY_OUTPUT}" \
    --sample-timeout ${SAMPLE_TIMEOUT} \
    --debug-trace \
    --print-turns
