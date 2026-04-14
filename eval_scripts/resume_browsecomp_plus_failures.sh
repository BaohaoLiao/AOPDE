#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
RESULTS_ROOT="${RESULTS_ROOT:-/data/agenthle/baohao/agentic_opd/openresearcher_results}"

SOURCE_DIR="${SOURCE_DIR:-${RESULTS_ROOT}/browsecomp_plus/OpenResearcher_dense_7x1}"
RETRY_OUTPUT_DIR="${RETRY_OUTPUT_DIR:-${SOURCE_DIR%/}_resume_failed}"
DATA_PATH="${DATA_PATH:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus/data/*.parquet}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0")

Seeds a fresh retry output directory with only successful qids from an existing
BrowseComp Plus run, then reruns failed or missing tasks via run_browsecomp_plus_full.sh.

Configurable environment variables:
  SOURCE_DIR        Existing run directory to inspect
                    Default: ${SOURCE_DIR}
  RETRY_OUTPUT_DIR  New output directory for the rerun
                    Default: ${RETRY_OUTPUT_DIR}
  DATA_PATH         BrowseComp Plus parquet glob

All service-related variables from run_browsecomp_plus_full.sh are also supported,
for example SEARCH_URL, MODEL_BASE_PORT, NUM_SERVERS, MODEL_PATH, SHOW_AGENT_LOGS.

Example:
  SOURCE_DIR=${RESULTS_ROOT}/browsecomp_plus/OpenResearcher_dense_7x1 \
  RETRY_OUTPUT_DIR=${RESULTS_ROOT}/browsecomp_plus/OpenResearcher_dense_7x1_retry \
  $(basename "$0")
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

require_glob_match() {
    local pattern="$1"
    compgen -G "$pattern" > /dev/null || fail "No files matched: ${pattern}"
}

seed_successes() {
    SOURCE_DIR="$SOURCE_DIR" RETRY_OUTPUT_DIR="$RETRY_OUTPUT_DIR" DATA_PATH="$DATA_PATH" /workspace/baohao/AOPDE/.eval/bin/python - <<'PY'
import glob
import json
import os
import sys
import duckdb

source_dir = os.environ["SOURCE_DIR"]
retry_output_dir = os.environ["RETRY_OUTPUT_DIR"]
data_path = os.environ["DATA_PATH"]

shard_files = glob.glob(os.path.join(source_dir, "node_*_shard_*.jsonl"))
if not shard_files:
    raise SystemExit(f"No shard files found in {source_dir}")

records = {}
for path in shard_files:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            qid = record.get("qid")
            if qid is None:
                continue
            records[qid] = record

success_records = [record for record in records.values() if record.get("status") == "success"]
failed_records = [record for record in records.values() if record.get("status") != "success"]

total_tasks = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{data_path}')").fetchone()[0]
missing = total_tasks - len(records)

print(f"total_tasks={total_tasks}")
print(f"source_records={len(records)}")
print(f"success_records={len(success_records)}")
print(f"failed_records={len(failed_records)}")
print(f"missing_records={missing}")

if len(failed_records) == 0 and missing == 0:
    raise SystemExit("Nothing to rerun: no failed or missing tasks detected.")

os.makedirs(retry_output_dir, exist_ok=True)
seed_path = os.path.join(retry_output_dir, "node_0_shard_seed.jsonl")
with open(seed_path, "w", encoding="utf-8") as handle:
    for record in sorted(success_records, key=lambda item: item["qid"]):
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

completed_qids_path = os.path.join(retry_output_dir, "completed_qids.txt")
with open(completed_qids_path, "w", encoding="utf-8") as handle:
    for record in sorted(success_records, key=lambda item: item["qid"]):
        handle.write(f"{record['qid']}\n")

print(f"seed_path={seed_path}")
print(f"completed_qids_path={completed_qids_path}")
PY
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    require_dir "$OPENRESEARCHER_DIR"
    require_dir "$SOURCE_DIR"
    require_glob_match "$DATA_PATH"

    if [[ -e "$RETRY_OUTPUT_DIR" ]] && [[ -n "$(find "$RETRY_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        fail "RETRY_OUTPUT_DIR already exists and is not empty: ${RETRY_OUTPUT_DIR}"
    fi

    echo "Preparing rerun directory"
    echo "  Source dir: ${SOURCE_DIR}"
    echo "  Retry output dir: ${RETRY_OUTPUT_DIR}"
    seed_successes

    echo "Launching rerun for failed or missing tasks"
    OUTPUT_DIR="$RETRY_OUTPUT_DIR" DATA_PATH="$DATA_PATH" "$SCRIPT_DIR/run_browsecomp_plus_full.sh"
}

main "$@"