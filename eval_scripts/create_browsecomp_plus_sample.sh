#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OPENRESEARCHER_DIR="${REPO_ROOT}/third_party/OpenResearcher"
EVAL_ENV_DIR="${REPO_ROOT}/.eval"
PYTHON_BIN="${EVAL_ENV_DIR}/bin/python"

FULL_DATA_GLOB="${FULL_DATA_GLOB:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus/data/*.parquet}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-0.20260414}"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-${OPENRESEARCHER_DIR}/Tevatron/browsecomp-plus-sample-${SAMPLE_SIZE}/data}"
SAMPLE_DATA_PATH="${SAMPLE_DATA_PATH:-${SAMPLE_DATA_DIR}/sample_${SAMPLE_SIZE}_seed_$(printf '%s' "${SAMPLE_SEED}" | tr '.' '_').parquet}"
SAMPLE_QUERY_IDS_PATH="${SAMPLE_QUERY_IDS_PATH:-${SAMPLE_DATA_PATH%.parquet}.query_ids.txt}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0")

Creates a reproducible random BrowseComp Plus parquet sample from the local shards.

Configurable environment variables:
  FULL_DATA_GLOB         Full BrowseComp Plus parquet glob
  SAMPLE_SIZE            Number of rows to sample without replacement (default: ${SAMPLE_SIZE})
  SAMPLE_SEED            DuckDB random seed in [0, 1) for reproducibility (default: ${SAMPLE_SEED})
  SAMPLE_DATA_DIR        Output directory for the sampled parquet
  SAMPLE_DATA_PATH       Output parquet path
  SAMPLE_QUERY_IDS_PATH  Output text file listing sampled query IDs

Examples:
  $(basename "$0")
  SAMPLE_SIZE=100 SAMPLE_SEED=0.5 $(basename "$0")
  SAMPLE_DATA_PATH=${OPENRESEARCHER_DIR}/Tevatron/my-sample/data/sample.parquet $(basename "$0")
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

validate_inputs() {
    [[ "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "SAMPLE_SIZE must be a positive integer"

    SAMPLE_SIZE="$SAMPLE_SIZE" SAMPLE_SEED="$SAMPLE_SEED" FULL_DATA_GLOB="$FULL_DATA_GLOB" "$PYTHON_BIN" - <<'PY'
import math
import os
import duckdb

sample_size = int(os.environ["SAMPLE_SIZE"])
sample_seed = float(os.environ["SAMPLE_SEED"])
source_glob = os.environ["FULL_DATA_GLOB"]

if not math.isfinite(sample_seed) or not (0.0 <= sample_seed < 1.0):
    raise SystemExit("SAMPLE_SEED must be a finite float in [0, 1)")

total_rows = duckdb.sql(
    f"SELECT COUNT(*) FROM read_parquet('{source_glob}')"
).fetchone()[0]

if sample_size > total_rows:
    raise SystemExit(
        f"SAMPLE_SIZE ({sample_size}) exceeds dataset size ({total_rows})"
    )
PY
}

create_sample_dataset() {
    mkdir -p "$SAMPLE_DATA_DIR"

    FULL_DATA_GLOB="$FULL_DATA_GLOB" \
    SAMPLE_SIZE="$SAMPLE_SIZE" \
    SAMPLE_SEED="$SAMPLE_SEED" \
    SAMPLE_DATA_PATH="$SAMPLE_DATA_PATH" \
    SAMPLE_QUERY_IDS_PATH="$SAMPLE_QUERY_IDS_PATH" \
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import duckdb

source_glob = os.environ["FULL_DATA_GLOB"]
sample_size = int(os.environ["SAMPLE_SIZE"])
sample_seed = float(os.environ["SAMPLE_SEED"])
output_path = Path(os.environ["SAMPLE_DATA_PATH"])
query_ids_path = Path(os.environ["SAMPLE_QUERY_IDS_PATH"])

con = duckdb.connect(database=':memory:')
con.execute("SELECT setseed(?)", [sample_seed])
con.execute(
    f"COPY ("
    f"SELECT * FROM read_parquet('{source_glob}') "
    f"ORDER BY random() LIMIT {sample_size}"
    f") TO '{output_path}' (FORMAT PARQUET)"
)

query_ids = con.execute(
    f"SELECT query_id FROM read_parquet('{output_path}') ORDER BY query_id"
).fetchall()
query_ids_path.write_text(
    "\n".join(row[0] for row in query_ids) + "\n",
    encoding="utf-8",
)

row_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
).fetchone()[0]
unique_query_ids = con.execute(
    f"SELECT COUNT(DISTINCT query_id) FROM read_parquet('{output_path}')"
).fetchone()[0]

print(output_path)
print(query_ids_path)
print(f"rows={row_count}")
print(f"unique_query_ids={unique_query_ids}")
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
    require_glob_match "$FULL_DATA_GLOB"

    validate_inputs

    echo "Creating BrowseComp Plus sample"
    echo "  Source glob: ${FULL_DATA_GLOB}"
    echo "  Sample size: ${SAMPLE_SIZE}"
    echo "  Sample seed: ${SAMPLE_SEED}"
    echo "  Output parquet: ${SAMPLE_DATA_PATH}"
    echo "  Query IDs: ${SAMPLE_QUERY_IDS_PATH}"

    create_sample_dataset
}

main "$@"
