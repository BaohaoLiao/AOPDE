# Retool + Search-R1 Multi-Teacher OPD

This top-level example trains one student on mixed Retool and Search-R1 prompts while routing OPD teacher logprobs per sample:

- `metadata.task == "retool"` uses the copied Retool rollout/reward code and the Retool teacher.
- `metadata.task == "search-r1"` uses the copied Search-R1 rollout/reward code and the Search-R1 teacher.

The folder is isolated: task implementations and helper files are copied from the top-level `retool/` and `search-r1/` folders.

## Prepare Mixed Data

```bash
python retool_and_search-r1/prepare_mixed_data.py \
  --retool-data /root/dapo-math-17k/dapo-math-17k.jsonl \
  --retool-input-key prompt \
  --retool-label-key label \
  --search-data /root/Search-R1/data/nq_hotpotqa_train/train.parquet \
  --search-input-key prompt \
  --search-label-key reward_model \
  --output /root/data/retool_search_mixed.jsonl
```

The output rows look like:

```json
{"prompt": "...", "label": "... or {...}", "metadata": {"task": "retool"}}
```

## Teacher Routing

Use external SGLang teacher servers:

```bash
export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate
```

Or use slime multi-model SGLang serving:

```bash
export SGLANG_CONFIG=/root/AOPDE/retool_and_search-r1/sglang_multiteacher.yaml
```

The default frozen model names are `retool_teacher` and `search_r1_teacher`. Override them with `RETOOL_TEACHER_MODEL_NAME` and `SEARCH_R1_TEACHER_MODEL_NAME`.

## Run

```bash
export PROMPT_DATA=/root/data/retool_search_mixed.jsonl
export HF_CHECKPOINT=/root/Qwen3-4B
export REF_LOAD=/root/Qwen3-4B_torch_dist
export SAVE_DIR=/root/Qwen3-4B_retool_search_opd

bash retool_and_search-r1/run_multiteacher_opd.sh
```

By default, `post_process_rewards()` returns the per-task scalar reward plus OPD. For pure distillation with zero task reward, set:

```bash
export MULTITEACHER_OPD_USE_TASK_REWARD=0
```

Main integration file:

- `generate_with_multiteacher_opd.py::generate()` routes rollout generation by `sample.metadata["task"]`.
- `generate_with_multiteacher_opd.py::reward_func()` computes the task reward and calls the selected teacher for token logprobs.
- `generate_with_multiteacher_opd.py::post_process_rewards()` stores `sample.teacher_log_probs` and preserves GRPO reward normalization behavior.
