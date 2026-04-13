#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="$REPO_ROOT/.eval/bin/activate"

if [[ ! -f "$ENV_PATH" ]]; then
  echo "Missing evaluation environment at $REPO_ROOT/.eval" >&2
  echo "Run scripts/setup_eval_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ENV_PATH"