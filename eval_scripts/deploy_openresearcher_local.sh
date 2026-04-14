#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
EVAL_ENV_DIR="${REPO_ROOT}/.eval"
PYTHON_BIN="${EVAL_ENV_DIR}/bin/python"
UVICORN_BIN="${EVAL_ENV_DIR}/bin/uvicorn"

DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-/data/agenthle/baohao/LLMs/Qwen/Qwen3-Embedding-8B}"
GENERATOR_MODEL_PATH="${GENERATOR_MODEL_PATH:-/data/agenthle/baohao/LLMs/OpenResearcher/OpenResearcher-30B-A3B}"
DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl}"
CORPUS_PARQUET_PATH="${CORPUS_PARQUET_PATH:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus-corpus/data/*.parquet}"
LUCENE_EXTRA_DIR="${LUCENE_EXTRA_DIR:-${OPENRESEARCHER_DIR}/tevatron}"

SEARCH_PORT="${SEARCH_PORT:-8000}"
RETRIEVER_GPUS="${RETRIEVER_GPUS:-7}"
MODEL_BASE_PORT="${MODEL_BASE_PORT:-8001}"
MODEL_GPUS="${MODEL_GPUS:-0,1,2,3,4,5,6}"
TP_SIZE="${TP_SIZE:-1}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") retriever
  $(basename "$0") model
  $(basename "$0") both

Default local model paths:
  Dense retriever: ${DENSE_MODEL_PATH}
  Generator model: ${GENERATOR_MODEL_PATH}

Configurable environment variables:
  DENSE_MODEL_PATH      Dense retriever model path
  GENERATOR_MODEL_PATH  OpenResearcher model path for vLLM
  DENSE_INDEX_PATH      BrowseComp Plus dense index glob
  CORPUS_PARQUET_PATH   BrowseComp Plus corpus parquet glob
  SEARCH_PORT           Dense retriever port (default: ${SEARCH_PORT})
  RETRIEVER_GPUS        CUDA_VISIBLE_DEVICES for retriever (default: ${RETRIEVER_GPUS})
  MODEL_BASE_PORT       First vLLM port (default: ${MODEL_BASE_PORT})
  MODEL_GPUS            CUDA_VISIBLE_DEVICES for vLLM servers (default: ${MODEL_GPUS})
  TP_SIZE               Tensor parallel size per vLLM server (default: ${TP_SIZE})

Examples:
  $(basename "$0") retriever
  MODEL_GPUS=0,1 TP_SIZE=2 $(basename "$0") model
  RETRIEVER_GPUS=6 MODEL_GPUS=0,1,2,3 $(basename "$0") both
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "Missing file: ${path}"
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] || fail "Missing directory: ${path}"
}

require_glob_match() {
    local pattern="$1"
    compgen -G "$pattern" > /dev/null || fail "No files matched: ${pattern}"
}

build_local_gpu_ids() {
    local visible_devices="$1"
    local ids=()
    local idx
    IFS=',' read -ra _gpu_array <<< "$visible_devices"

    for idx in "${!_gpu_array[@]}"; do
        ids+=("$idx")
    done

    local IFS=','
    echo "${ids[*]}"
}

ensure_common_prereqs() {
    require_dir "$OPENRESEARCHER_DIR"
    require_dir "$EVAL_ENV_DIR"
    require_file "$PYTHON_BIN"
    require_file "$UVICORN_BIN"
    require_dir "$LUCENE_EXTRA_DIR"
    require_file "${LUCENE_EXTRA_DIR}/lucene-highlighter-9.9.1.jar"
    require_glob_match "$CORPUS_PARQUET_PATH"
}

ensure_retriever_prereqs() {
    ensure_common_prereqs
    require_glob_match "$DENSE_INDEX_PATH"
    require_dir "$DENSE_MODEL_PATH"
}

ensure_model_prereqs() {
    ensure_common_prereqs
    require_dir "$GENERATOR_MODEL_PATH"
}

start_retriever() {
    local gpu_ids

    ensure_retriever_prereqs
    gpu_ids="$(build_local_gpu_ids "$RETRIEVER_GPUS")"

    cd "$OPENRESEARCHER_DIR"
    export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPUS"
    export GPU_IDS="$gpu_ids"
    export SEARCHER_TYPE="dense"
    export DENSE_INDEX_PATH="$DENSE_INDEX_PATH"
    export DENSE_MODEL_NAME="$DENSE_MODEL_PATH"
    export CORPUS_PARQUET_PATH="$CORPUS_PARQUET_PATH"
    export LUCENE_EXTRA_DIR="$LUCENE_EXTRA_DIR"

    echo "Starting dense retriever"
    echo "  Model: ${DENSE_MODEL_NAME}"
    echo "  Index: ${DENSE_INDEX_PATH}"
    echo "  Port: ${SEARCH_PORT}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "  GPU_IDS: ${GPU_IDS}"

    exec "$UVICORN_BIN" scripts.deploy_search_service:app --host 0.0.0.0 --port "$SEARCH_PORT"
}

start_model() {
    ensure_model_prereqs

    cd "$OPENRESEARCHER_DIR"
    echo "Starting OpenResearcher vLLM servers"
    echo "  Model: ${GENERATOR_MODEL_PATH}"
    echo "  Base port: ${MODEL_BASE_PORT}"
    echo "  TP size: ${TP_SIZE}"
    echo "  CUDA_VISIBLE_DEVICES: ${MODEL_GPUS}"

    exec bash scripts/start_nemotron_servers.sh "$TP_SIZE" "$MODEL_BASE_PORT" "$MODEL_GPUS" "$GENERATOR_MODEL_PATH"
}

start_both() {
    local retriever_log
    local retriever_pid
    local gpu_ids

    ensure_retriever_prereqs
    ensure_model_prereqs
    gpu_ids="$(build_local_gpu_ids "$RETRIEVER_GPUS")"

    cd "$OPENRESEARCHER_DIR"
    mkdir -p logs
    retriever_log="logs/dense_retriever_${SEARCH_PORT}.log"

    (
        export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPUS"
        export GPU_IDS="$gpu_ids"
        export SEARCHER_TYPE="dense"
        export DENSE_INDEX_PATH="$DENSE_INDEX_PATH"
        export DENSE_MODEL_NAME="$DENSE_MODEL_PATH"
        export CORPUS_PARQUET_PATH="$CORPUS_PARQUET_PATH"
        export LUCENE_EXTRA_DIR="$LUCENE_EXTRA_DIR"
        exec "$UVICORN_BIN" scripts.deploy_search_service:app --host 0.0.0.0 --port "$SEARCH_PORT"
    ) > "$retriever_log" 2>&1 &

    retriever_pid=$!
    trap 'kill "$retriever_pid" 2>/dev/null || true' EXIT INT TERM

    sleep 3
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
        cat "$retriever_log" >&2
        fail "Dense retriever failed to start. See ${retriever_log}"
    fi

    echo "Dense retriever is running in background"
    echo "  PID: ${retriever_pid}"
    echo "  Log: ${OPENRESEARCHER_DIR}/${retriever_log}"

    bash scripts/start_nemotron_servers.sh "$TP_SIZE" "$MODEL_BASE_PORT" "$MODEL_GPUS" "$GENERATOR_MODEL_PATH"
}

case "${1:-help}" in
    retriever)
        start_retriever
        ;;
    model)
        start_model
        ;;
    both)
        start_both
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
