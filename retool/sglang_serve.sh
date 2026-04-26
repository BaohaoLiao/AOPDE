#!/bin/bash
# Launch a stand-alone SGLang server suitable for retool/eval.py.
#
# Usage:
#   bash retool/sglang_serve.sh
#
# All knobs are env-overridable, e.g.:
#   MODEL_PATH=/path/to/hf PORT=30000 TP_SIZE=2 \
#       CUDA_VISIBLE_DEVICES=0,1 bash retool/sglang_serve.sh
#
# Notes:
#   * The server runs in the foreground. Background it with `&` if needed.
#   * `cd /tmp` avoids `PermissionError: [Errno 13] '/home/jovyan'` that hits
#     SGLang's TP worker subprocesses (multiprocessing spawn-mode chdir).
#   * USER/LOGNAME/HOME defaults work around UIDs missing from /etc/passwd
#     (e.g. container UID 19001), which otherwise breaks
#     `getpass.getuser()` inside torch._inductor.

set -ex

MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to an HF checkpoint directory"}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
EXTRA_ARGS=${EXTRA_ARGS:-""}

# Defensive env so SGLang/torch don't crash on missing pwd entries.
export USER=${USER:-user}
export LOGNAME=${LOGNAME:-$USER}
export HOME=${HOME:-/tmp}

# Move to a directory the current UID can chdir into. SGLang spawns TP workers
# via multiprocessing "spawn"; each child does os.chdir(parent_cwd).
cd /tmp

CMD=(
    python3 -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --tp "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
    --trust-remote-code
)

if [ "${DISABLE_RADIX_CACHE}" = "1" ]; then
    CMD+=(--disable-radix-cache)
fi

if [ -n "${EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    CMD+=(${EXTRA_ARGS})
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
