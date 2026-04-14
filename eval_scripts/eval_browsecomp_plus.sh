#!/usr/bin/env bash

set -euo pipefail

source /workspace/baohao/AOPDE/.eval/bin/activate

/workspace/baohao/AOPDE/.eval/bin/python /workspace/baohao/AOPDE/eval_scripts/eval_with_openai_compatible.py \
  --input_dir /data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/MiniMax_M2.7_mini_sample100 \
  --output_file /data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/MiniMax_M2.7_mini_sample100/eval_results.json \
  --base_url http://localhost:4141/v1 \
  --model gpt-4.1 \
  --api_key dummy