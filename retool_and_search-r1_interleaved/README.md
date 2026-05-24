# Retool + Search-R1 Interleaved Multi-Teacher OPD

This top-level example trains one student on Retool and Search-R1 prompts while keeping each rollout/training batch task-homogeneous:

- Rollout 0 uses only Retool prompts.
- Rollout 1 uses only Search-R1 prompts.
- The pattern repeats according to `INTERLEAVED_TASK_ORDER`, default `retool,search-r1`.
- Retool batches use the copied Retool rollout/reward code and the Retool teacher.
- Search-R1 batches use the copied Search-R1 rollout/reward code and the Search-R1 teacher.

The folder is isolated: task implementations and helper files are copied from the top-level `retool/` and `search-r1/` folders.

## Input Data

Use two separate prompt files. They may be `.jsonl` or `.parquet`, matching Slime's normal dataset loader.

Retool defaults:

```bash
export RETOOL_PROMPT_DATA=/root/data/retool.jsonl
export RETOOL_INPUT_KEY=prompt
export RETOOL_LABEL_KEY=label
export RETOOL_METADATA_KEY=metadata
```

Search-R1 defaults:

```bash
export SEARCH_R1_PROMPT_DATA=/root/data/search_r1.parquet
export SEARCH_R1_INPUT_KEY=prompt
export SEARCH_R1_LABEL_KEY=reward_model
export SEARCH_R1_METADATA_KEY=metadata
```

The interleaved data source injects `metadata.task` internally, so the two input files do not need to be pre-mixed.

## Teacher Routing

Use external SGLang teacher servers:

```bash
export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate
```

Or use slime multi-model SGLang serving:

```bash
export SGLANG_CONFIG=/root/AOPDE/retool_and_search-r1_interleaved/sglang_multiteacher.yaml
```

The default frozen model names are `retool_teacher` and `search_r1_teacher`. Override them with `RETOOL_TEACHER_MODEL_NAME` and `SEARCH_R1_TEACHER_MODEL_NAME`.

## Run

```bash
export RETOOL_PROMPT_DATA=/root/data/retool.jsonl
export SEARCH_R1_PROMPT_DATA=/root/data/search_r1.parquet
export HF_CHECKPOINT=/root/Qwen3-4B
export REF_LOAD=/root/Qwen3-4B_torch_dist
export SAVE_DIR=/root/Qwen3-4B_retool_search_interleaved_opd

bash retool_and_search-r1_interleaved/run_multiteacher_opd.sh
```

To start with Search-R1 instead of Retool or to change the pattern:

```bash
export INTERLEAVED_TASK_ORDER=search-r1,retool
```

By default, `post_process_rewards()` returns the per-task scalar reward plus OPD. For pure distillation with zero task reward, set:

```bash
export MULTITEACHER_OPD_USE_TASK_REWARD=0
```

Main integration file:

- `generate_with_multiteacher_opd.py::generate()` routes rollout generation by `sample.metadata["task"]`.
- `generate_with_multiteacher_opd.py::reward_func()` computes the task reward and calls the selected teacher for token logprobs.
- `generate_with_multiteacher_opd.py::post_process_rewards()` stores `sample.teacher_log_probs` and preserves GRPO reward normalization behavior.
- `interleaved_data_source.py::InterleavedTaskDataSource` loads the two prompt files and returns only one task per rollout batch.
- `interleaved_rollout.py::generate_rollout()` selects the active task from the rollout id.
