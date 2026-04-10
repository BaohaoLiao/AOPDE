# OpenResearcher Evaluation Setup

This document describes the evaluation-only environment used by AOPDE when running benchmarks with the OpenResearcher submodule.

## Scope

- Use this setup for evaluation workflows only.
- Keep training dependencies separate from evaluation dependencies.
- Do not download large benchmark assets unless they are needed for the run.

## Repository Layout Assumption

This repository tracks OpenResearcher as a submodule at `third_party/OpenResearcher`.

Run from the main repository root:

```bash
cd /path/to/AOPDE
git submodule update --init --recursive
```

## Goal

Create a reproducible Python `3.12` environment at `.eval/` with the dependencies required to run OpenResearcher evaluation.

## Prerequisites

- Linux environment with `apt-get`
- Network access
- Enough disk space for large Python packages such as `vllm`

## Recommended Setup

Use the repo-provided setup script:

```bash
./scripts/setup_eval_env.sh
```

That script will:

- install or reuse `uv`
- ensure Python `3.12`
- ensure Java `21`
- create `.eval/`
- clone `tevatron` into `third_party/OpenResearcher/tevatron` if missing
- install `tevatron` and `OpenResearcher` editable into `.eval`
- run a basic verification step

## Manual Equivalent

If you need to reproduce the setup manually, the equivalent flow is:

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

## Daily Use

```bash
cd /path/to/AOPDE
source .eval/bin/activate
```

or:

```bash
source scripts/activate_eval.sh
```

## Notes

- This setup is sufficient for evaluating existing checkpoints.
- Benchmark downloads remain optional and may be large.
- If benchmark assets are required, run `bash third_party/OpenResearcher/setup.sh` after activating `.eval`.
- `.eval/` should remain untracked.

