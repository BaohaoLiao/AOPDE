#!/bin/bash

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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

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
source "./scripts/models/qwen3-4B.sh"

MODEL_DIR=/data/agenthle/baohao/agentic_opd/retool/model
DATA_DIR=/data/agenthle/baohao/agentic_opd/retool/data
ROLLOUT_DEBUG_DIR=${MODEL_DIR}/qwen3-4b-sft-rl/rollout_debug


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/qwen3-4b-sft
   --ref-load ${MODEL_DIR}/qwen3-4b-sft_torch_dist
   # --load /root/Qwen3-4B_slime/
   --save ${MODEL_DIR}/qwen3-4b-sft-rl/
   --save-interval 20
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
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

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime  ${DATA_DIR}/aime-2024/aime-2024.jsonl
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
   --wandb-group qwen3-4B-sft-rl
   --wandb-key ${WANDB_KEY}
)

TRAIN_ENV_ARGS=(
   --train-env-vars "{\"NVTE_FLASH_ATTN\":\"${NVTE_FLASH_ATTN:-1}\",\"NVTE_FUSED_ATTN\":\"${NVTE_FUSED_ATTN:-1}\",\"NVTE_UNFUSED_ATTN\":\"${NVTE_UNFUSED_ATTN:-0}\",\"NVTE_DEBUG\":\"${NVTE_DEBUG:-0}\",\"NVTE_DEBUG_LEVEL\":\"${NVTE_DEBUG_LEVEL:-0}\"}"
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
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# launch the master node of ray in container
export CUDA_VISIBLE_DEVICES=6,7
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
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
   ${TRAIN_ENV_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
