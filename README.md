# AOPDE

This repository is the main entrypoint for training and evaluation. OpenResearcher is tracked as a submodule under `third_party/OpenResearcher` and is used for BrowseComp Plus evaluation.

## Scope

- Use the `.eval` environment for evaluation workflows.
- Keep training dependencies separate from evaluation dependencies.
- Do not install evaluation-only packages into the training environment.
- For AOPDE evaluation, use `.eval` rather than `third_party/OpenResearcher/.venv`.

## Repository Layout

Relevant paths:

- `install_scripts/setup_eval_env.sh`: creates the evaluation environment
- `install_scripts/activate_eval.sh`: activates `.eval`
- `eval_scripts/deploy_openresearcher_local.sh`: starts the local dense retriever and OpenResearcher vLLM server
- `eval_scripts/run_browsecomp_plus_smoke_test.sh`: runs a one-example BrowseComp Plus smoke test
- `third_party/OpenResearcher`: OpenResearcher submodule

Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

## Prerequisites

- Linux environment with `apt-get`
- network access for initial model and dataset downloads
- enough disk space for Python dependencies and BrowseComp Plus assets
- GPUs for the retriever and vLLM model server

## Installation

Create the dedicated evaluation environment from the main repository root:

```bash
./install_scripts/setup_eval_env.sh
```

This script will:

- install or reuse `uv`
- ensure Python `3.12`
- ensure Java `21`
- create `.eval/`
- clone `tevatron` into `third_party/OpenResearcher/tevatron` if missing
- install `tevatron` and `OpenResearcher` editable into `.eval`
- run a basic verification step

Activate the environment with either of these:

```bash
source .eval/bin/activate
```

or:

```bash
source install_scripts/activate_eval.sh
```

For day-to-day evaluation in this repository, `.eval` is the primary environment. The helper scripts under `third_party/OpenResearcher` are configured to prefer `.eval` and only fall back to `third_party/OpenResearcher/.venv` when OpenResearcher is used standalone outside AOPDE.

## Manual Setup

If you need to reproduce the environment manually, use:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env

uv python install 3.12

apt-get update
apt-get install -y openjdk-21-jdk

uv venv .eval --python 3.12
source .eval/bin/activate

if [ ! -d third_party/OpenResearcher/tevatron ]; then
	git clone https://github.com/texttron/tevatron.git third_party/OpenResearcher/tevatron
fi

uv pip install -e third_party/OpenResearcher/tevatron
uv pip install -e third_party/OpenResearcher
```

## Verification

Run these checks from the repo root after activation:

```bash
source .eval/bin/activate
python --version
java -version
python -c "import vllm, datasets, pyserini, duckdb; print('imports_ok')"
python third_party/OpenResearcher/eval.py --help
```

Expected results:

- Python reports `3.12.x`
- Java reports `21.x`
- `imports_ok` prints successfully
- `eval.py --help` runs without import errors

## BrowseComp Plus Asset Setup

After `.eval` is ready, prepare the BrowseComp Plus benchmark assets:

```bash
source .eval/bin/activate
cd third_party/OpenResearcher
bash ./setup.sh
```

Verified behavior of `third_party/OpenResearcher/setup.sh` in this repository:

- uses `/workspace/baohao/AOPDE/.eval` when it exists
- reinstalls `OpenResearcher` and `tevatron` editable into `.eval`
- downloads Lucene jars into `third_party/OpenResearcher/tevatron`
- downloads benchmark assets into `third_party/OpenResearcher/Tevatron`

Expected asset directories after setup:

- `third_party/OpenResearcher/Tevatron/browsecomp-plus`
- `third_party/OpenResearcher/Tevatron/browsecomp-plus-corpus`
- `third_party/OpenResearcher/Tevatron/browsecomp-plus-indexes/bm25`
- `third_party/OpenResearcher/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b`

Approximate disk usage for BrowseComp Plus assets:

- dataset: `2.6G`
- corpus: `1.7G`
- indexes: `3.6G`

## Local Model Paths

The helper scripts in `eval_scripts` default to these local model paths:

- dense retriever: `/data/agenthle/baohao/LLMs/Qwen/Qwen3-Embedding-8B`
- generator model: `/data/agenthle/baohao/LLMs/OpenResearcher/OpenResearcher-30B-A3B`
- result root: `/data/agenthle/baohao/agentic_opd/openresearcher_results`

These defaults can be overridden through environment variables when needed.

## BrowseComp Plus Smoke Test

The quickest end-to-end test is:

1. Start the dense retriever service.
2. Start the OpenResearcher vLLM server.
3. Run the one-example smoke test.

From the repo root, in terminal 1:

```bash
source .eval/bin/activate
./eval_scripts/deploy_openresearcher_local.sh retriever
```

In terminal 2:

```bash
source .eval/bin/activate
./eval_scripts/deploy_openresearcher_local.sh model
```

In terminal 3:

```bash
source .eval/bin/activate
./eval_scripts/run_browsecomp_plus_smoke_test.sh
```

The smoke test script:

- creates a one-row parquet slice at `third_party/OpenResearcher/Tevatron/browsecomp-plus-smoke/data/smoke.parquet`
- runs `deploy_agent.py` against the local search service at `http://localhost:8000`
- runs `deploy_agent.py` against the local vLLM server at `http://localhost:8001/v1`

If you want to force a fresh smoke test run and avoid reusing an old output directory:

```bash
source .eval/bin/activate
OUTPUT_DIR=/data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/smoke_test_$(date +%Y%m%d_%H%M%S) ./eval_scripts/run_browsecomp_plus_smoke_test.sh
```

Common overrides:

```bash
RETRIEVER_GPUS=6 MODEL_GPUS=0,1,2,3 TP_SIZE=2 ./eval_scripts/deploy_openresearcher_local.sh both
```

```bash
SEARCH_URL=http://localhost:8000 VLLM_SERVER_URL=http://localhost:8001/v1 ./eval_scripts/run_browsecomp_plus_smoke_test.sh
```

## Full BrowseComp Plus Evaluation

For a full run, start the retriever and vLLM services the same way, then run the agent on the full dataset.

Terminal 1:

```bash
source .eval/bin/activate
./eval_scripts/deploy_openresearcher_local.sh retriever
```

Terminal 2:

```bash
source .eval/bin/activate
./eval_scripts/deploy_openresearcher_local.sh model
```

Terminal 3:

```bash
source .eval/bin/activate
./eval_scripts/run_browsecomp_plus_full.sh
```

By default this writes to:

- `/data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/OpenResearcher_dense_7x1`

To create a fresh timestamped output directory:

```bash
source .eval/bin/activate
OUTPUT_DIR=/data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/run_$(date +%Y%m%d_%H%M%S) ./eval_scripts/run_browsecomp_plus_full.sh
```

## Result Evaluation

After a run completes, evaluate the outputs with:

```bash
source .eval/bin/activate
/workspace/baohao/AOPDE/.eval/bin/python third_party/OpenResearcher/eval.py --input_dir /data/agenthle/baohao/agentic_opd/openresearcher_results/browsecomp_plus/OpenResearcher_dense_7x1
```

Replace the input directory with your actual run directory when needed.

## Notes

- This setup is sufficient for evaluating existing checkpoints.
- Large benchmark downloads are optional and are not part of `install_scripts/setup_eval_env.sh`.
- If benchmark assets are required, run `bash third_party/OpenResearcher/setup.sh` after activating `.eval`.
- Install any additional evaluation Python packages into `.eval`, not `third_party/OpenResearcher/.venv`.
- `.eval/` should remain untracked.