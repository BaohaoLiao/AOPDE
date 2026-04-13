#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
EVAL_ENV_DIR="${REPO_ROOT}/.eval"
PYTHON_BIN="${EVAL_ENV_DIR}/bin/python"

SEARCH_URL="${SEARCH_URL:-http://localhost:8000}"
VLLM_SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8001/v1}"
MODEL_PATH="${MODEL_PATH:-/data/agenthle/baohao/LLMs/OpenResearcher/OpenResearcher-30B-A3B}"
FULL_DATA_GLOB="${FULL_DATA_GLOB:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus/data/*.parquet}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus-smoke/data}"
SMOKE_DATA_PATH="${SMOKE_DATA_PATH:-${SMOKE_DATA_DIR}/smoke.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPENRESEARCHER_DIR}/results/browsecomp_plus/smoke_test}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0")

This script:
  1. creates a one-example BrowseComp Plus parquet slice
  2. runs deploy_agent.py against the local search service and local vLLM server

Required running services:
  - dense retriever on ${SEARCH_URL}
  - vLLM OpenAI-compatible server on ${VLLM_SERVER_URL}

Configurable environment variables:
  SEARCH_URL        Search service URL (default: ${SEARCH_URL})
  VLLM_SERVER_URL   vLLM API URL (default: ${VLLM_SERVER_URL})
  MODEL_PATH        Local OpenResearcher model path (default: ${MODEL_PATH})
  FULL_DATA_GLOB    Full BrowseComp Plus parquet glob
  SMOKE_DATA_PATH   Output parquet for the one-example smoke dataset
  OUTPUT_DIR        Output directory for smoke test results
  MAX_CONCURRENCY   Max concurrency per worker (default: ${MAX_CONCURRENCY})
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] || fail "Missing directory: ${path}"
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "Missing file: ${path}"
}

require_glob_match() {
    local pattern="$1"
    compgen -G "$pattern" > /dev/null || fail "No files matched: ${pattern}"
}

create_smoke_dataset() {
    mkdir -p "$SMOKE_DATA_DIR"

    FULL_DATA_GLOB="$FULL_DATA_GLOB" SMOKE_DATA_PATH="$SMOKE_DATA_PATH" "$PYTHON_BIN" - <<'PY'
import os
import duckdb

source_glob = os.environ["FULL_DATA_GLOB"]
output_path = os.environ["SMOKE_DATA_PATH"]

con = duckdb.connect(database=':memory:')
con.execute(
    f"COPY (SELECT * FROM read_parquet('{source_glob}') ORDER BY query_id LIMIT 1) "
    f"TO '{output_path}' (FORMAT PARQUET)"
)
print(output_path)
PY
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    require_dir "$OPENRESEARCHER_DIR"
    require_dir "$EVAL_ENV_DIR"
    require_file "$PYTHON_BIN"
    require_dir "$MODEL_PATH"
    require_glob_match "$FULL_DATA_GLOB"

    create_smoke_dataset
    require_file "$SMOKE_DATA_PATH"

    mkdir -p "$OUTPUT_DIR"

    cd "$OPENRESEARCHER_DIR"
    exec "$PYTHON_BIN" deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL_PATH" \
        --search_url "$SEARCH_URL" \
        --dataset_name browsecomp_plus \
        --data_path "$SMOKE_DATA_PATH" \
        --browser_backend local \
        --reasoning_effort high \
        --vllm_server_url "$VLLM_SERVER_URL" \
        --max_concurrency_per_worker "$MAX_CONCURRENCY"
}

main "$@"