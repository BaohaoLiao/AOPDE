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

MODEL_DIR=${MODEL_DIR:-/mount/coreai-genai-pvc/baliao/experiments/agentic_opd/00_single_env/search_r1}
DATA_DIR=${DATA_DIR:-${MODEL_DIR}/data}

HF_CHECKPOINT=${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000154/hf}
REF_LOAD=${REF_LOAD:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT/iter_0000154/torch_dist}
SAVE_DIR=${SAVE_DIR:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-n1-interleaved-search-r1}
ROLLOUT_DEBUG_DIR=${SAVE_DIR}/rollout_debug

SEARCH_R1_PROMPT_DATA=${SEARCH_R1_PROMPT_DATA:-${DATA_DIR}/nq_hotpotqa_train/train_reformat_6.5k.parquet}
SEARCH_R1_INPUT_KEY=${SEARCH_R1_INPUT_KEY:-prompt}
SEARCH_R1_LABEL_KEY=${SEARCH_R1_LABEL_KEY:-reward_model}
SEARCH_R1_METADATA_KEY=${SEARCH_R1_METADATA_KEY:-metadata}
INTERLEAVED_TASK_ORDER=${INTERLEAVED_TASK_ORDER:-search-r1}

rm -rf "${SAVE_DIR}"
mkdir -p "${ROLLOUT_DEBUG_DIR}"

# ---------------------------------------------------------------------------
# Search-R1 teacher model server (external to Ray, uses GPUs 4-7 by default)
# ---------------------------------------------------------------------------
START_SEARCH_R1_TEACHER=${START_SEARCH_R1_TEACHER:-1}
SEARCH_R1_TEACHER_MODEL_PATH=${SEARCH_R1_TEACHER_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B-Instruct-2507-SFT-RL/iter_0000199/hf}
SEARCH_R1_TEACHER_GPUS=${SEARCH_R1_TEACHER_GPUS:-4,5,6,7}
SEARCH_R1_TEACHER_TP=${SEARCH_R1_TEACHER_TP:-1}
SEARCH_R1_TEACHER_DP=${SEARCH_R1_TEACHER_DP:-4}
SEARCH_R1_TEACHER_PORT=${SEARCH_R1_TEACHER_PORT:-13141}
SEARCH_R1_TEACHER_MEM_FRACTION_STATIC=${SEARCH_R1_TEACHER_MEM_FRACTION_STATIC:-0.80}
SEARCH_R1_TEACHER_CHUNKED_PREFILL_SIZE=${SEARCH_R1_TEACHER_CHUNKED_PREFILL_SIZE:-2048}
SEARCH_R1_TEACHER_MAX_RUNNING_REQUESTS=${SEARCH_R1_TEACHER_MAX_RUNNING_REQUESTS:-128}
SEARCH_R1_TEACHER_MAX_TOTAL_TOKENS=${SEARCH_R1_TEACHER_MAX_TOTAL_TOKENS:-16384}
SEARCH_R1_TEACHER_SCHEDULE_CONSERVATIVENESS=${SEARCH_R1_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.5}
SEARCH_R1_TEACHER_EXTRA_ARGS=${SEARCH_R1_TEACHER_EXTRA_ARGS:-}
SEARCH_R1_TEACHER_PID=""

if [ "${START_SEARCH_R1_TEACHER}" = "1" ]; then
   SEARCH_R1_TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
   SEARCH_R1_TEACHER_IP="${SEARCH_R1_TEACHER_IP:-127.0.0.1}"
   SEARCH_R1_TEACHER_URL="http://${SEARCH_R1_TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/generate"
   SEARCH_R1_TEACHER_LOG="${SAVE_DIR}/sglang_search_r1_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   CUDA_VISIBLE_DEVICES=${SEARCH_R1_TEACHER_GPUS} \
      bash -c "cd /tmp && exec python3 -m sglang.launch_server \
      --model-path '${SEARCH_R1_TEACHER_MODEL_PATH}' \
      --host 0.0.0.0 \
      --port '${SEARCH_R1_TEACHER_PORT}' \
      --tp '${SEARCH_R1_TEACHER_TP}' \
      --dp '${SEARCH_R1_TEACHER_DP}' \
      --chunked-prefill-size '${SEARCH_R1_TEACHER_CHUNKED_PREFILL_SIZE}' \
      --mem-fraction-static '${SEARCH_R1_TEACHER_MEM_FRACTION_STATIC}' \
      --max-running-requests '${SEARCH_R1_TEACHER_MAX_RUNNING_REQUESTS}' \
      --max-total-tokens '${SEARCH_R1_TEACHER_MAX_TOTAL_TOKENS}' \
      --schedule-conservativeness '${SEARCH_R1_TEACHER_SCHEDULE_CONSERVATIVENESS}' \
      --disable-radix-cache \
      ${SEARCH_R1_TEACHER_EXTRA_ARGS}" \
      > "${SEARCH_R1_TEACHER_LOG}" 2>&1 &
   SEARCH_R1_TEACHER_PID=$!

   cleanup_teacher() {
      if [ -n "${SEARCH_R1_TEACHER_PID:-}" ]; then
         kill "${SEARCH_R1_TEACHER_PID}" 2>/dev/null || true
      fi
   }
   trap cleanup_teacher EXIT

   echo "Starting Search-R1 teacher model server..."
   until curl -sf --noproxy '*' "http://${SEARCH_R1_TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Search-R1 teacher server..."
      tail -n 10 "${SEARCH_R1_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${SEARCH_R1_TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/get_model_info"
   echo "Search-R1 teacher server is up at ${SEARCH_R1_TEACHER_URL}."
   sleep 10
else
   SEARCH_R1_TEACHER_URL=${SEARCH_R1_TEACHER_URL:-http://127.0.0.1:13141/generate}
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
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --opd-sft-coef "${OPD_SFT_COEF:-0.0}"
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
   --weight-decay "${WEIGHT_DECAY:-0.01}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
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
WANDB_GROUP=${WANDB_GROUP:-SearchR1_Qwen3-4B-Instruct-2507-SFT-RL-to-Qwen3-4B-Instruct-2507-SFT-n1-interleaved-search-r1}
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
export SEARCH_URL=${SEARCH_URL:-"http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve"}
export SEARCH_R1_MAX_TURNS=${SEARCH_R1_MAX_TURNS:-4}
export MULTITEACHER_OPD_USE_TASK_REWARD=${MULTITEACHER_OPD_USE_TASK_REWARD:-1}

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
    \"SEARCH_URL\": \"${SEARCH_URL}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS}\",
    \"SEARCH_R1_TEACHER_URL\": \"${SEARCH_R1_TEACHER_URL}\",
    \"SEARCH_R1_PROMPT_DATA\": \"${SEARCH_R1_PROMPT_DATA}\",
    \"SEARCH_R1_INPUT_KEY\": \"${SEARCH_R1_INPUT_KEY}\",
    \"SEARCH_R1_LABEL_KEY\": \"${SEARCH_R1_LABEL_KEY}\",
    \"SEARCH_R1_METADATA_KEY\": \"${SEARCH_R1_METADATA_KEY}\",
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
