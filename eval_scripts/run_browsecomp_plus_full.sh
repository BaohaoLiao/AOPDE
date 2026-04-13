#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
EVAL_ENV_DIR="${REPO_ROOT}/.eval"
PYTHON_BIN="${EVAL_ENV_DIR}/bin/python"

SEARCH_URL="${SEARCH_URL:-http://localhost:8000}"
MODEL_BASE_PORT="${MODEL_BASE_PORT:-8001}"
NUM_SERVERS="${NUM_SERVERS:-7}"
MODEL_PATH="${MODEL_PATH:-/data/agenthle/baohao/LLMs/OpenResearcher/OpenResearcher-30B-A3B}"
DATA_PATH="${DATA_PATH:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus/data/*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPENRESEARCHER_DIR}/results/browsecomp_plus/OpenResearcher_dense_7x1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
BROWSER_BACKEND="${BROWSER_BACKEND:-local}"
CHECK_SERVICES="${CHECK_SERVICES:-1}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0")

Runs the full BrowseComp Plus evaluation against already-deployed local services.

Default assumptions:
  - search service: ${SEARCH_URL}
  - vLLM replicas: http://localhost:${MODEL_BASE_PORT}/v1 through http://localhost:$((MODEL_BASE_PORT + NUM_SERVERS - 1))/v1
  - number of replicas: ${NUM_SERVERS}
  - model path: ${MODEL_PATH}
  - output dir: ${OUTPUT_DIR}

Configurable environment variables:
  SEARCH_URL        Search service URL (default: ${SEARCH_URL})
  MODEL_BASE_PORT   First vLLM port (default: ${MODEL_BASE_PORT})
  NUM_SERVERS       Number of deployed vLLM replicas (default: ${NUM_SERVERS})
  MODEL_PATH        Local OpenResearcher model path (default: ${MODEL_PATH})
  DATA_PATH         BrowseComp Plus parquet glob
  OUTPUT_DIR        Output directory for result shards
  MAX_CONCURRENCY   Max concurrency per worker (default: ${MAX_CONCURRENCY})
  BROWSER_BACKEND   Browser backend (default: ${BROWSER_BACKEND})
  CHECK_SERVICES    1 to verify endpoints before running, 0 to skip

Examples:
  $(basename "$0")
  OUTPUT_DIR=${OPENRESEARCHER_DIR}/results/browsecomp_plus/run_$(date +%Y%m%d_%H%M%S) $(basename "$0")
  MODEL_BASE_PORT=8101 NUM_SERVERS=7 $(basename "$0")
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

build_server_urls() {
    local urls=()
    local idx

    for ((idx = 0; idx < NUM_SERVERS; idx += 1)); do
        urls+=("http://localhost:$((MODEL_BASE_PORT + idx))/v1")
    done

    local IFS=,
    echo "${urls[*]}"
}

check_services() {
    local server_urls="$1"

    SEARCH_URL="$SEARCH_URL" SERVER_URLS="$server_urls" "$PYTHON_BIN" - <<'PY'
import os
import sys
import urllib.request

search_url = os.environ["SEARCH_URL"].rstrip("/")
server_urls = [url.strip().rstrip("/") for url in os.environ["SERVER_URLS"].split(",") if url.strip()]

def check(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

check(search_url)
for base_url in server_urls:
    check(f"{base_url}/models")

print("service_checks_ok")
PY
}

main() {
    local server_urls

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    require_dir "$OPENRESEARCHER_DIR"
    require_dir "$EVAL_ENV_DIR"
    require_file "$PYTHON_BIN"
    require_dir "$MODEL_PATH"
    require_glob_match "$DATA_PATH"

    if ! [[ "$NUM_SERVERS" =~ ^[1-9][0-9]*$ ]]; then
        fail "NUM_SERVERS must be a positive integer"
    fi

    if ! [[ "$MODEL_BASE_PORT" =~ ^[0-9]+$ ]]; then
        fail "MODEL_BASE_PORT must be an integer"
    fi

    server_urls="$(build_server_urls)"

    if [[ "$CHECK_SERVICES" == "1" ]]; then
        check_services "$server_urls"
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "Running full BrowseComp Plus evaluation"
    echo "  Search service: ${SEARCH_URL}"
    echo "  vLLM servers: ${server_urls}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Model path: ${MODEL_PATH}"
    echo "  Data path: ${DATA_PATH}"

    cd "$OPENRESEARCHER_DIR"
    exec "$PYTHON_BIN" deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL_PATH" \
        --search_url "$SEARCH_URL" \
        --dataset_name browsecomp_plus \
        --data_path "$DATA_PATH" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$server_urls" \
        --max_concurrency_per_worker "$MAX_CONCURRENCY"
}

main "$@"