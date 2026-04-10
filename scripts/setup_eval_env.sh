#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENRESEARCHER_DIR="$REPO_ROOT/third_party/OpenResearcher"
TEVATRON_DIR="$OPENRESEARCHER_DIR/tevatron"
UV_ENV_FILE="/root/.local/bin/env"
OPENRESEARCHER_GIT_DIR="$(git -C "$OPENRESEARCHER_DIR" rev-parse --git-dir)"

if [[ ! -d "$OPENRESEARCHER_DIR" ]]; then
  echo "Missing OpenResearcher submodule at $OPENRESEARCHER_DIR" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required for Java installation." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ -f "$UV_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$UV_ENV_FILE"
fi

uv python install 3.12

if ! command -v java >/dev/null 2>&1 || ! java -version 2>&1 | grep -q '21'; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-21-jdk
fi

cd "$REPO_ROOT"
uv venv .eval --python 3.12 --clear

# shellcheck disable=SC1091
source "$REPO_ROOT/.eval/bin/activate"

if [[ ! -d "$TEVATRON_DIR" ]]; then
  git clone https://github.com/texttron/tevatron.git "$TEVATRON_DIR"
fi

mkdir -p "$OPENRESEARCHER_GIT_DIR/info"
if ! grep -qx 'tevatron/' "$OPENRESEARCHER_GIT_DIR/info/exclude" 2>/dev/null; then
  echo 'tevatron/' >> "$OPENRESEARCHER_GIT_DIR/info/exclude"
fi

uv pip install -e "$TEVATRON_DIR"
uv pip install -e "$OPENRESEARCHER_DIR"

python --version
java -version
python -c "import vllm, datasets, pyserini, duckdb; print('imports_ok')"
python "$OPENRESEARCHER_DIR/eval.py" --help >/dev/null

cat <<EOF
Evaluation environment is ready.
Activate it with:
  source "$REPO_ROOT/.eval/bin/activate"
EOF