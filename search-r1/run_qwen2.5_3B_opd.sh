#!/bin/bash

# Search-R1 OPD v2 launch template.
# The teacher runs as a separate SGLang server (sglang mode), started by this
# script by default. Set START_TEACHER=0 and TEACHER_URL=... to use an
# externally managed teacher.
# This script is kept separate from run_qwen2.5_3B.sh so regular Search-R1
# runs keep using the original generate/reward code.

# for rerun the task. Do not kill arbitrary Python by default because the
# retriever is often a Python process. Set KILL_EXISTING_PROCESSES=1 only when
# you want the aggressive cleanup behavior.
if [ "${KILL_EXISTING_PROCESSES:-0}" = "1" ]; then
   pkill -9 sglang
   sleep 3
   ray stop --force
   pkill -9 ray
   pkill -9 python
   sleep 3
   pkill -9 ray
   pkill -9 python
else
   ray stop --force || true
fi

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp}"
export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
source "${SCRIPT_DIR}/../scripts/models/qwen2.5-3B.sh"

# Comma-separated local retriever URLs. Override from the shell if needed.
SEARCH_URL=${SEARCH_URL:-"http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve"}

# Teacher model configuration. Override these for a larger teacher.
START_TEACHER=${START_TEACHER:-"1"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"/root/Qwen2.5-3B/"}
TEACHER_TP=${TEACHER_TP:-"4"}
TEACHER_GPUS=${TEACHER_GPUS:-"4,5,6,7"}
TEACHER_PORT=${TEACHER_PORT:-"13141"}
TEACHER_PID=""

# Student/Ray GPU configuration. Keep this disjoint from TEACHER_GPUS when
# START_TEACHER=1.
STUDENT_GPUS=${STUDENT_GPUS:-"0,1,2,3"}
STUDENT_RAY_GPUS=${STUDENT_RAY_GPUS:-"4"}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-"${STUDENT_RAY_GPUS}"}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-"${STUDENT_RAY_GPUS}"}

if [ "${START_TEACHER}" = "1" ]; then
   TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
   TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
   TEACHER_URL="http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   TEACHER_LOG="/tmp/sglang_search_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
      bash -c "cd /tmp && exec python3 -m sglang.launch_server \
      --model-path ${TEACHER_MODEL_PATH} \
      --host 0.0.0.0 \
      --port ${TEACHER_PORT} \
      --tp ${TEACHER_TP} \
      --chunked-prefill-size 2048 \
      --mem-fraction-static ${TEACHER_MEM_FRACTION_STATIC:-0.80} \
      --max-running-requests ${TEACHER_MAX_RUNNING_REQUESTS:-32} \
      --max-total-tokens ${TEACHER_MAX_TOTAL_TOKENS:-65536} \
      --schedule-conservativeness ${TEACHER_SCHEDULE_CONSERVATIVENESS:-0.5} \
      --disable-radix-cache" \
      > "${TEACHER_LOG}" 2>&1 &
   TEACHER_PID=$!

   cleanup_teacher() {
      if [ -n "${TEACHER_PID:-}" ]; then
         kill "${TEACHER_PID}" 2>/dev/null || true
      fi
   }
   trap cleanup_teacher EXIT

   echo "Starting Search-R1 OPD teacher model server..."
   until curl -sf --noproxy '*' "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for teacher server..."
      tail -n 10 "${TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info"
   echo "Teacher server is up at ${TEACHER_URL}."
   sleep 10
else
   TEACHER_URL=${TEACHER_URL:-"http://127.0.0.1:13141/generate"}
fi

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-3B/
   --ref-load /root/Qwen2.5-3B_torch_dist/
   # --load /root/Qwen2.5-3B_slime/
   # --save /root/Qwen2.5-3B_slime/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 512
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef ${OPD_KL_COEF:-"1.0"}
   --opd-sft-coef ${OPD_SFT_COEF:-"0.0"}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group search-r1_qwen2.5-3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path opd_generate_with_search.generate
   --custom-rm-path on_policy_distillation_v2.reward_func
   --custom-reward-post-process-path on_policy_distillation_v2.post_process_rewards
   --rm-url ${TEACHER_URL}

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES=${STUDENT_GPUS}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${STUDENT_RAY_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SEARCH_URL\": \"${SEARCH_URL}\",
    \"SEARCH_OPD_MIX_PROMPT_ASSISTANT\": \"${SEARCH_OPD_MIX_PROMPT_ASSISTANT:-0}\",
    \"SEARCH_OPD_REV_TURN_DECAY\": \"${SEARCH_OPD_REV_TURN_DECAY:-0.8}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"USER\": \"${USER}\",
    \"LOGNAME\": \"${LOGNAME}\",
    \"HOME\": \"${HOME}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPD_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
