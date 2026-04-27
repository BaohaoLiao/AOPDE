#!/bin/bash

# On-policy distillation for retool.
# The teacher runs as a separate SGLang server (sglang mode).
# The student is trained with GRPO + OPD KL penalty.
#
# Usage:
#   bash retool/retool_opd.sh
#
# Adjust TEACHER_MODEL_PATH, STUDENT_* paths, TEACHER_TP, and GPU assignments
# to match your hardware.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

# Some container UIDs (e.g. 19001) are not in /etc/passwd, which makes
# getpass.getuser() -> pwd.getpwuid() raise KeyError inside torch._inductor.
# Set USER/LOGNAME so getpass falls back to env vars instead of pwd lookup.
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp}"

# Bypass any corporate HTTP proxy for local sglang servers — otherwise
# sglang's internal startup self-check (and curl below) routes
# http://0.0.0.0:13141/model_info through the proxy, times out, and the
# server process gets killed.
export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-/opt/Megatron-LM}"
SLIME_ROOT="${SLIME_ROOT:-/opt/slime}"

# ---------------------------------------------------------------------------
# Paths — edit these
# ---------------------------------------------------------------------------
MODEL_DIR=/data/agenthle/baohao/agentic_opd/retool/model
DATA_DIR=/data/agenthle/baohao/agentic_opd/retool/data
ROLLOUT_DEBUG_DIR=${MODEL_DIR}/retool-opd/rollout_debug

# Teacher model (larger / stronger, runs on SGLang server)
TEACHER_MODEL_PATH=${MODEL_DIR}/teacher-model-hf   # e.g. Qwen3-32B HF checkpoint
TEACHER_TP=4                                        # tensor-parallel size for teacher server
TEACHER_GPUS="4,5,6,7"                             # which GPUs host the teacher

# Student model (smaller, trained by slime)
STUDENT_HF_CKPT=${MODEL_DIR}/qwen3-4b-sft          # HF checkpoint used to seed sglang rollout engine
STUDENT_TORCH_DIST=${MODEL_DIR}/qwen3-4b-sft_torch_dist
STUDENT_SAVE=${MODEL_DIR}/retool-opd/

# ---------------------------------------------------------------------------
# Start teacher SGLang server on dedicated GPUs
# ---------------------------------------------------------------------------
# Use the node's routable IP, not 127.0.0.1 — Ray rollout workers may run on
# other nodes and 127.0.0.1 on those nodes won't reach the teacher.
TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT=13141
TEACHER_LOG="/tmp/sglang_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
    bash -c "cd /tmp && exec python3 -m sglang.launch_server \
    --model-path ${TEACHER_MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp ${TEACHER_TP} \
    --chunked-prefill-size 2048 \
    --mem-fraction-static 0.80 \
    --max-running-requests 32 \
    --max-total-tokens 65536 \
    --schedule-conservativeness 0.5 \
    --disable-radix-cache" \
    > "${TEACHER_LOG}" 2>&1 &

echo "Starting teacher model server..."
until curl -sf --noproxy '*' http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
    echo "Waiting for teacher server..."
    tail -n 10 "${TEACHER_LOG}"
    sleep 5
done
curl --noproxy '*' http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info
echo "Teacher server is up at ${TEACHER_IP}:${TEACHER_PORT}."
sleep 10

# ---------------------------------------------------------------------------
# Student model architecture args (qwen3-1.7B)
# ---------------------------------------------------------------------------
source "${SCRIPT_DIR}/../scripts/models/qwen3-1.7B.sh"

# ---------------------------------------------------------------------------
# Training args
# ---------------------------------------------------------------------------
CKPT_ARGS=(
   --hf-checkpoint ${STUDENT_HF_CKPT}
   --ref-load ${STUDENT_TORCH_DIST}
   --save ${STUDENT_SAVE}
   --save-interval 20
   --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
   --save-debug-rollout-data ${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt
   --use-dynamic-global-batch-size
)

# Custom generate uses retool tool execution; OPD reward gets teacher log-probs
# and logs math_dapo score for monitoring (not used as reward).
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path retool.on_policy_distillation.reward_func
   --custom-reward-post-process-path retool.on_policy_distillation.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project agentic-opd
   --wandb-group retool-opd
   --wandb-key ${WANDB_KEY}
)

# Don't force NVTE_*_ATTN here. Megatron's `auto` attention backend asserts
# that any pre-set NVTE_{FLASH,FUSED,UNFUSED}_ATTN matches its expected
# values; passing explicit 0/1 mismatches and crashes init. Leave them unset
# (or use --attention-backend to pin a specific backend).
TRAIN_ENV_ARGS=(
   --train-env-vars "{\"NVTE_DEBUG\":\"${NVTE_DEBUG:-0}\",\"NVTE_DEBUG_LEVEL\":\"${NVTE_DEBUG_LEVEL:-0}\"}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---------------------------------------------------------------------------
# Launch Ray and submit training job (student uses GPUs 0-3)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-"subprocess"}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-"32"}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-"300"}
export TOOL_SANDBOX_MAX_TURNS=${TOOL_SANDBOX_MAX_TURNS:-"16"}
export TOOL_SANDBOX_MAX_TOOL_CALLS=${TOOL_SANDBOX_MAX_TOOL_CALLS:-"16"}

# Raise file-descriptor and process limits. Defaults of 1024/4096 in many
# containers will get exhausted by 32+ tool subprocesses (each holds 6+ fds)
# combined with 64+ aiohttp sockets to the teacher and 256+ to the student
# router. FD exhaustion in uvloop manifests as a SIGABRT in the asyncio loop
# thread.
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[opd] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"TOOL_SANDBOX_BACKEND\": \"${TOOL_SANDBOX_BACKEND}\",
    \"TOOL_SANDBOX_JUPYTER_TIMEOUT\": \"${TOOL_SANDBOX_JUPYTER_TIMEOUT}\",
    \"TOOL_SANDBOX_MAX_TURNS\": \"${TOOL_SANDBOX_MAX_TURNS}\",
    \"TOOL_SANDBOX_MAX_TOOL_CALLS\": \"${TOOL_SANDBOX_MAX_TOOL_CALLS}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}\",
    \"USER\": \"${USER}\",
    \"LOGNAME\": \"${LOGNAME}\",
    \"HOME\": \"${HOME}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPD_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${TRAIN_ENV_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
