#!/bin/bash

set -ex

unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

export PYTHONBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp}"

export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_DIR}/third_party/slime"
MEGATRON_ROOT="${MEGATRON_ROOT:-/opt/Megatron-LM}"

MODEL_DIR=${MODEL_DIR:-/mount/coreai-genai-pvc/baliao/experiments/agentic_opd/00_single_env/retool}
DATA_DIR=${DATA_DIR:-${MODEL_DIR}/data}

HF_CHECKPOINT=${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000154/hf}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000154/torch_dist}
SAVE_DIR=${SAVE_DIR:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-n1-interleaved-retool}
ROLLOUT_DEBUG_DIR=${SAVE_DIR}/rollout_debug

RETOOL_PROMPT_DATA=${RETOOL_PROMPT_DATA:-${DATA_DIR}/dapo-math-6.4k/dapo-math-6.4k.jsonl}
RETOOL_INPUT_KEY=${RETOOL_INPUT_KEY:-prompt}
RETOOL_LABEL_KEY=${RETOOL_LABEL_KEY:-label}
RETOOL_METADATA_KEY=${RETOOL_METADATA_KEY:-metadata}
INTERLEAVED_TASK_ORDER=${INTERLEAVED_TASK_ORDER:-retool}

rm -rf "${SAVE_DIR}"
mkdir -p "${ROLLOUT_DEBUG_DIR}"

# ---------------------------------------------------------------------------
# Retool teacher model server (external to Ray, uses GPUs 4-7 by default)
# ---------------------------------------------------------------------------
START_RETOOL_TEACHER=${START_RETOOL_TEACHER:-1}
RETOOL_TEACHER_MODEL_PATH=${RETOOL_TEACHER_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL/iter_0000199/hf}
RETOOL_TEACHER_GPUS=${RETOOL_TEACHER_GPUS:-4,5,6,7}
RETOOL_TEACHER_TP=${RETOOL_TEACHER_TP:-1}
RETOOL_TEACHER_DP=${RETOOL_TEACHER_DP:-4}
RETOOL_TEACHER_PORT=${RETOOL_TEACHER_PORT:-13141}
RETOOL_TEACHER_MEM_FRACTION_STATIC=${RETOOL_TEACHER_MEM_FRACTION_STATIC:-0.85}
RETOOL_TEACHER_CHUNKED_PREFILL_SIZE=${RETOOL_TEACHER_CHUNKED_PREFILL_SIZE:-2048}
RETOOL_TEACHER_MAX_RUNNING_REQUESTS=${RETOOL_TEACHER_MAX_RUNNING_REQUESTS:-128}
RETOOL_TEACHER_SCHEDULE_CONSERVATIVENESS=${RETOOL_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.3}
RETOOL_TEACHER_EXTRA_ARGS=${RETOOL_TEACHER_EXTRA_ARGS:-}
RETOOL_TEACHER_PID=""

if [ "${START_RETOOL_TEACHER}" = "1" ]; then
   RETOOL_TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
   RETOOL_TEACHER_IP="${RETOOL_TEACHER_IP:-127.0.0.1}"
   RETOOL_TEACHER_URL="http://${RETOOL_TEACHER_IP}:${RETOOL_TEACHER_PORT}/generate"
   RETOOL_TEACHER_LOG="${SAVE_DIR}/sglang_retool_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   CUDA_VISIBLE_DEVICES=${RETOOL_TEACHER_GPUS} \
      bash -c "cd /tmp && exec python3 -m sglang.launch_server \
      --model-path '${RETOOL_TEACHER_MODEL_PATH}' \
      --host 0.0.0.0 \
      --port '${RETOOL_TEACHER_PORT}' \
      --tp '${RETOOL_TEACHER_TP}' \
      --dp '${RETOOL_TEACHER_DP}' \
      --chunked-prefill-size '${RETOOL_TEACHER_CHUNKED_PREFILL_SIZE}' \
      --mem-fraction-static '${RETOOL_TEACHER_MEM_FRACTION_STATIC}' \
      --max-running-requests '${RETOOL_TEACHER_MAX_RUNNING_REQUESTS}' \
      --schedule-conservativeness '${RETOOL_TEACHER_SCHEDULE_CONSERVATIVENESS}' \
      --disable-radix-cache \
      ${RETOOL_TEACHER_EXTRA_ARGS}" \
      > "${RETOOL_TEACHER_LOG}" 2>&1 &
   RETOOL_TEACHER_PID=$!

   cleanup_teacher() {
      if [ -n "${RETOOL_TEACHER_PID:-}" ]; then
         kill "${RETOOL_TEACHER_PID}" 2>/dev/null || true
      fi
   }
   trap cleanup_teacher EXIT

   echo "Starting Retool teacher model server..."
   until curl -sf --noproxy '*' "http://${RETOOL_TEACHER_IP}:${RETOOL_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Retool teacher server..."
      tail -n 10 "${RETOOL_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${RETOOL_TEACHER_IP}:${RETOOL_TEACHER_PORT}/get_model_info"
   echo "Retool teacher server is up at ${RETOOL_TEACHER_URL}."
   sleep 10
else
   RETOOL_TEACHER_URL=${RETOOL_TEACHER_URL:-http://127.0.0.1:13141/generate}
fi

# ---------------------------------------------------------------------------
# Student model architecture args
# ---------------------------------------------------------------------------
MODEL_CONFIG=${MODEL_CONFIG:-"${SLIME_DIR}/scripts/models/qwen3-4B.sh"}
source "${MODEL_CONFIG}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --load "${SAVE_DIR}"
   --save-interval 20
   --rotary-base "${ROTARY_BASE:-5000000}"
)

ROLLOUT_ARGS=(
   --data-source-path interleaved_data_source.InterleavedTaskDataSource
   --rollout-function-path interleaved_rollout.generate_rollout
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-256}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
   --use-dynamic-global-batch-size
   --rollout-stop "\\</tool_call>"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.5}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_multiteacher_opd.generate
   --custom-rm-path generate_with_multiteacher_opd.reward_func
   --custom-reward-post-process-path generate_with_multiteacher_opd.post_process_rewards
)

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-ReTool_Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-n1-interleaved-retool}
DISABLE_WANDB_RANDOM_SUFFIX=${DISABLE_WANDB_RANDOM_SUFFIX:-1}
export WANDB_DIR=${WANDB_DIR:-${SAVE_DIR}}

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
   WANDB_ARGS+=(--use-wandb)
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
   WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
   WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi
if [[ "${DISABLE_WANDB_RANDOM_SUFFIX}" == "1" ]]; then
   WANDB_ARGS+=(--disable-wandb-random-suffix)
fi

# ---------------------------------------------------------------------------
# Launch Ray and submit training job (student uses GPUs 0-3 by default)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${STUDENT_GPUS:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-subprocess}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-128}
export TOOL_SANDBOX_MAX_TURNS=${TOOL_SANDBOX_MAX_TURNS:-16}
export TOOL_SANDBOX_MAX_TOOL_CALLS=${TOOL_SANDBOX_MAX_TOOL_CALLS:-16}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-300}
export MULTITEACHER_OPD_USE_TASK_REWARD=${MULTITEACHER_OPD_USE_TASK_REWARD:-0}

RAY_NUM_GPUS=${RAY_NUM_GPUS:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}

ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[opd] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"TOOL_SANDBOX_BACKEND\": \"${TOOL_SANDBOX_BACKEND}\",
    \"TOOL_SANDBOX_CONCURRENCY\": \"${TOOL_SANDBOX_CONCURRENCY}\",
    \"TOOL_SANDBOX_MAX_TURNS\": \"${TOOL_SANDBOX_MAX_TURNS}\",
    \"TOOL_SANDBOX_MAX_TOOL_CALLS\": \"${TOOL_SANDBOX_MAX_TOOL_CALLS}\",
    \"TOOL_SANDBOX_JUPYTER_TIMEOUT\": \"${TOOL_SANDBOX_JUPYTER_TIMEOUT}\",
    \"RETOOL_TEACHER_URL\": \"${RETOOL_TEACHER_URL}\",
    \"RETOOL_PROMPT_DATA\": \"${RETOOL_PROMPT_DATA}\",
    \"RETOOL_INPUT_KEY\": \"${RETOOL_INPUT_KEY}\",
    \"RETOOL_LABEL_KEY\": \"${RETOOL_LABEL_KEY}\",
    \"RETOOL_METADATA_KEY\": \"${RETOOL_METADATA_KEY}\",
    \"INTERLEAVED_TASK_ORDER\": \"${INTERLEAVED_TASK_ORDER}\",
    \"MULTITEACHER_OPD_USE_TASK_REWARD\": \"${MULTITEACHER_OPD_USE_TASK_REWARD}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"USER\": \"${USER}\",
    \"LOGNAME\": \"${LOGNAME}\",
    \"HOME\": \"${HOME}\",
    \"no_proxy\": \"${no_proxy:-}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_DIR}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
