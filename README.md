# AOPDE

This repository is the main entrypoint for training and evaluating the model.

## Evaluation Environment

OpenResearcher is tracked as a submodule at `third_party/OpenResearcher` and is used for evaluation.

Initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Create the dedicated evaluation environment from the main repo root:

```bash
./scripts/setup_eval_env.sh
```

This script will:

- install or reuse `uv`
- ensure Python `3.12`
- ensure Java `21`
- create `.eval/`
- clone `tevatron` inside `third_party/OpenResearcher` if needed
- install `tevatron` and `OpenResearcher` into `.eval`

Activate the evaluation environment with either of these:

```bash
source .eval/bin/activate
```

or:

```bash
source scripts/activate_eval.sh
```

## Notes

- The evaluation environment is isolated from training dependencies.
- Large benchmark downloads are optional and are not part of `setup_eval_env.sh`.
- If you need benchmark assets for OpenResearcher, run `bash third_party/OpenResearcher/setup.sh` after activating `.eval`.