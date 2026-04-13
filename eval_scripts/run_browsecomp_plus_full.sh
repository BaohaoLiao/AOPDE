#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
EVAL_ENV_DIR="${REPO_ROOT}/.eval"
PYTHON_BIN="${EVAL_ENV_DIR}/bin/python"
RESULTS_ROOT="${RESULTS_ROOT:-/data/agenthle/baohao/agentic_opd/openresearcher_results}"

SEARCH_URL="${SEARCH_URL:-http://localhost:8000}"
MODEL_BASE_PORT="${MODEL_BASE_PORT:-8001}"
NUM_SERVERS="${NUM_SERVERS:-7}"
MODEL_PATH="${MODEL_PATH:-/data/agenthle/baohao/LLMs/OpenResearcher/OpenResearcher-30B-A3B}"
DATA_PATH="${DATA_PATH:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus/data/*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/browsecomp_plus/OpenResearcher_dense_7x1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
BROWSER_BACKEND="${BROWSER_BACKEND:-local}"
CHECK_SERVICES="${CHECK_SERVICES:-1}"
SHOW_AGENT_LOGS="${SHOW_AGENT_LOGS:-0}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-15}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/run.log}"

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
    RESULTS_ROOT      Root directory for evaluation outputs (default: ${RESULTS_ROOT})
  MODEL_PATH        Local OpenResearcher model path (default: ${MODEL_PATH})
  DATA_PATH         BrowseComp Plus parquet glob
  OUTPUT_DIR        Output directory for result shards
  MAX_CONCURRENCY   Max concurrency per worker (default: ${MAX_CONCURRENCY})
  BROWSER_BACKEND   Browser backend (default: ${BROWSER_BACKEND})
  CHECK_SERVICES    1 to verify endpoints before running, 0 to skip
    SHOW_AGENT_LOGS   1 to stream deploy_agent logs to terminal, 0 to write them to RUN_LOG only
    MONITOR_INTERVAL_S  Seconds between progress updates (default: ${MONITOR_INTERVAL_S})
    RUN_LOG           Log file for deploy_agent output (default: ${RUN_LOG})

Examples:
  $(basename "$0")
    OUTPUT_DIR=${RESULTS_ROOT}/browsecomp_plus/run_$(date +%Y%m%d_%H%M%S) $(basename "$0")
  MODEL_BASE_PORT=8101 NUM_SERVERS=7 $(basename "$0")
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
    fi

    if [[ -n "${agent_pid:-}" ]] && kill -0 "$agent_pid" 2>/dev/null; then
        kill "$agent_pid" 2>/dev/null || true
    fi
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

resolve_total_tasks() {
    DATA_PATH="$DATA_PATH" "$PYTHON_BIN" - <<'PY'
import os
import duckdb

pattern = os.environ["DATA_PATH"]
count = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchone()[0]
print(count)
PY
}

progress_snapshot() {
    OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON_BIN" - <<'PY'
import glob
import json
import os

output_dir = os.environ["OUTPUT_DIR"]
records = {}

for shard_file in glob.glob(os.path.join(output_dir, "node_*_shard_*.jsonl")):
    with open(shard_file, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            qid = record.get("qid")
            if qid is None:
                continue
            records[qid] = record.get("status")

completed = len(records)
success = sum(1 for status in records.values() if status == "success")
failed = sum(1 for status in records.values() if status == "fail")
print(f"{completed} {success} {failed}")
PY
}

monitor_progress() {
    local total_tasks="$1"
    local completed success failed remaining

    while kill -0 "$agent_pid" 2>/dev/null; do
        read -r completed success failed < <(progress_snapshot)
        remaining=$((total_tasks - completed))
        printf '[progress] completed %s/%s | success %s | fail %s | remaining %s\n' \
            "$completed" "$total_tasks" "$success" "$failed" "$remaining"
        sleep "$MONITOR_INTERVAL_S"
    done
}

main() {
    local server_urls total_tasks completed success failed remaining

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
    total_tasks="$(resolve_total_tasks)"

    if ! [[ "$MONITOR_INTERVAL_S" =~ ^[1-9][0-9]*$ ]]; then
        fail "MONITOR_INTERVAL_S must be a positive integer"
    fi

    echo "Running full BrowseComp Plus evaluation"
    echo "  Search service: ${SEARCH_URL}"
    echo "  vLLM servers: ${server_urls}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Model path: ${MODEL_PATH}"
    echo "  Data path: ${DATA_PATH}"
    echo "  Total tasks: ${total_tasks}"
    echo "  Log file: ${RUN_LOG}"

    cd "$OPENRESEARCHER_DIR"
    trap cleanup EXIT INT TERM

    if [[ "$SHOW_AGENT_LOGS" == "1" ]]; then
        "$PYTHON_BIN" deploy_agent.py \
            --output_dir "$OUTPUT_DIR" \
            --model_name_or_path "$MODEL_PATH" \
            --search_url "$SEARCH_URL" \
            --dataset_name browsecomp_plus \
            --data_path "$DATA_PATH" \
            --browser_backend "$BROWSER_BACKEND" \
            --reasoning_effort high \
            --vllm_server_url "$server_urls" \
            --max_concurrency_per_worker "$MAX_CONCURRENCY" \
            2>&1 | tee -a "$RUN_LOG" &
    else
        "$PYTHON_BIN" deploy_agent.py \
            --output_dir "$OUTPUT_DIR" \
            --model_name_or_path "$MODEL_PATH" \
            --search_url "$SEARCH_URL" \
            --dataset_name browsecomp_plus \
            --data_path "$DATA_PATH" \
            --browser_backend "$BROWSER_BACKEND" \
            --reasoning_effort high \
            --vllm_server_url "$server_urls" \
            --max_concurrency_per_worker "$MAX_CONCURRENCY" \
            > "$RUN_LOG" 2>&1 &
    fi

    agent_pid=$!
    monitor_progress "$total_tasks" &
    monitor_pid=$!

    wait "$agent_pid"

    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
    fi

    read -r completed success failed < <(progress_snapshot)
    remaining=$((total_tasks - completed))
    printf '[progress] completed %s/%s | success %s | fail %s | remaining %s\n' \
        "$completed" "$total_tasks" "$success" "$failed" "$remaining"
    echo "Evaluation finished. Logs: ${RUN_LOG}"
}

main "$@"