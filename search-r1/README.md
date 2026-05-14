# Search-R1 lite

This is a minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and an example of using multi-turn conversation and tool-calling in slime.

## Environment Setup

Use the `slimerl/slime:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e . --no-deps
# for Search R1
pip install chardet
```

Download and prepare the training data:

```bash
cd /root/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
pip install -e . --no-deps
pip install tensordict

# Set your working directory
WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# Process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# (Optional) Process multiple dataset search format test file
# Note: the final file is not shuffled
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**Note:** If you plan to use local search backend, see the [Appendix](#appendix-setting-up-local-retriever) for instructions on setting up the local retrieval server.

Initialize the Qwen2.5-3B model:

```bash
# hf checkpoint
hf download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

## Configuration

### Search Backend Configuration

The `generate_with_search.py` file supports both **local search** and **Google search** backends. Configure via the `SEARCH_R1_CONFIGS` dictionary:

```python
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,

    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"

    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,
    },

    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual serper.dev API key
        "snippet_only": True,
        "proxy": None,
    },

    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities (required for TIS)

    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}
```

#### Using Local Search

1. Set `"search_backend": "local"`
2. Configure `"local"` section with your local retrieval server URL
3. Start your local search server before running the training script

#### Using Google Search

1. Set `"search_backend": "google"`
2. Configure `"google"` section with your serper.dev API key
3. Get your API key from [serper.dev](https://serper.dev)

### Enabling TIS (Trajectory Importance Sampling)

TIS requires log probability collection. To enable TIS:

**1. In `generate_with_search.py`:**
```python
SEARCH_R1_CONFIGS = {
    # ... other configs
    "return_logprob": True,  # Must be True for TIS
}
```

**2. In `run_qwen2.5_3B.sh`:**

Uncomment the TIS-related arguments in `GRPO_ARGS`:
```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # Uncomment to enable TIS
   --use-tis
)
```

And uncomment the TIS configuration paths in `CUSTOM_ARGS`:
```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # Uncomment to enable TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

**Important Notes:**
- TIS requires `return_logprob=True` in `SEARCH_R1_CONFIGS`
- When collecting log probabilities, response postprocessing is automatically disabled to maintain token/logp alignment
- TIS adds computational overhead but can improve training efficiency

## Running the Script

```bash
cd slime/
bash examples/search-r1/run_qwen2.5_3B.sh
```

## Code Structure

To implement multi-turn conversation + tool-calling in slime, you only need to implement a custom data generation function and a reward model for the task. These correspond to the following 2 configuration items in the startup script:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

These are the `generate` and `reward_func` functions in `generate_with_search.py`.

## Evaluation

`search-r1/eval.py` runs an async, multi-turn rollout (`<search>` / `<answer>`) against a stand-alone SGLang inference server plus the local retrieval server, and scores responses with the Search-R1 EM grader. Three terminals are involved:

1. Retrieval server (the `retriever` env from the Appendix)
2. SGLang inference server (your training/inference env, with sglang installed)
3. Eval driver

### 1. Start the retrieval server

Follow the [Appendix](#appendix-setting-up-local-retriever). Once it's up on `http://127.0.0.1:8000/retrieve`, leave it running.

### 2. Start the SGLang server

In a separate shell (NOT the `retriever` env), use `search-r1/sglang_serve.sh`. All knobs are env-overridable:

```bash
MODEL_PATH=/path/to/hf_checkpoint \
PORT=30000 \
TP_SIZE=2 DP_SIZE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
    bash search-r1/sglang_serve.sh
```

Total GPUs used = `TP_SIZE * DP_SIZE`; set `CUDA_VISIBLE_DEVICES` accordingly. Wait until you see SGLang's `The server is fired up and ready to roll!` log line.

### 3. Run the eval driver

In a third shell:

```bash
MODEL_PATH=/path/to/hf_checkpoint \
DATASET=/path/to/test.parquet \
PORT=30000 \
SEARCH_URL=http://127.0.0.1:8000/retrieve \
    bash search-r1/eval.sh
```

Required env:
- `MODEL_PATH` — the same HF checkpoint dir SGLang is serving (used for the tokenizer / chat template).
- `DATASET` — Search-R1 test parquet (verl-style schema with `prompt`, `data_source`, `reward_model.ground_truth.target`). Also accepts `.jsonl` or an HF dataset id.

Common optional env:
- `MAX_TURNS` (default `4`), `TOPK` (default `3`)
- `NUM_SAMPLES` (default `1`) — n samples per prompt
- `MAX_CONCURRENT` (default `8`) — concurrent rollouts
- `LIMIT` — only evaluate the first N rows
- `DATA_SOURCE_FILTER` — comma-separated list of `data_source` values to keep (e.g. `nq,hotpotqa`)
- `TEMPERATURE` (default `0.0`), `TOP_P` (default `1.0`)
- `MAX_NEW_TOKENS` (default `1024`), `MAX_CONTEXT_LEN` (default `8192`)
- `OUTPUT` / `SUMMARY_OUTPUT` — JSONL + summary paths (default under `$MODEL_PATH/`)
- `PRINT_TURNS=1` — print each turn to stdout for sanity-checking
- `DEBUG_TRACE=1` — verbose request/response logging

The driver writes one JSON line per (prompt, sample) including `rendered_prompt`, the multi-turn trace, the predicted answer, the gold target, and the EM score; the summary file aggregates per-`data_source` accuracy.

## Appendix: Setting up Local Retriever

This section provides detailed instructions for setting up the local dense retriever for use with the local search backend.

### Prerequisites

The local retriever requires a separate conda environment to avoid conflicts with the training environment. It uses GPU for efficient retrieval.

### Step 1: Install Conda

If you don't have conda installed, run the following commands:

```bash
# Download and install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Step 2: Create Retriever Environment

Create and activate a conda environment with Python 3.10:

```bash
# Create environment
# conda create -n retriever python=3.10 -y
# conda activate retriever

mkdir -p /mount/coreai-genai-pvc/baliao/home/.cache/conda
mkdir -p /mount/coreai-genai-pvc/baliao/home/.conda/pkgs
mkdir -p /mount/coreai-genai-pvc/baliao/home/.conda/envs

export HOME=/mount/coreai-genai-pvc/baliao/home
export XDG_CACHE_HOME=/mount/coreai-genai-pvc/baliao/home/.cache
export CONDA_PKGS_DIRS=/mount/coreai-genai-pvc/baliao/home/.conda/pkgs
export CONDA_ENVS_PATH=/mount/coreai-genai-pvc/baliao/home/.conda/envs

conda create -p /mount/coreai-genai-pvc/baliao/home/.conda/envs/retriever python=3.10 -y
conda activate /mount/coreai-genai-pvc/baliao/home/.conda/envs/retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Required packages. Pin transformers to a torch-2.4-compatible version
# (newer transformers calls torch APIs that don't exist in 2.4 and will
# raise `infer_schema(...) Parameter input has unsupported type torch.Tensor`).
pip install "transformers==4.46.3" datasets pyserini huggingface_hub
pip install uvicorn fastapi

# torchvision is not used by the retrieval server and the conda-installed
# torchvision often mismatches torch (causing
# `RuntimeError: operator torchvision::nms does not exist`).
# Easiest: just remove it.
pip uninstall -y torchvision torchaudio
```

#### Install faiss with GPU Python bindings

The PyPI wheel (`faiss-gpu-cu12`) is convenient but only ships kernels for SM 7.0–8.9 — it will crash on Hopper (H100, SM 9.0) with `CUDA error 209 no kernel image is available for execution on the device`. The conda-forge package on many clusters ships a C++-only build (no `site-packages/faiss/`) or one without GPU symbols. The reliable path on H100 is to **build faiss v1.9.0 from source**:

```bash
# Build deps
conda install -y -c conda-forge cmake "swig=4.2.*" mkl mkl-devel

# Source
cd /path/to/work_dir   # your build location
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.9.0    # v1.8.0's swig file is incompatible with modern swig

# Configure (BUILD_TESTING=OFF skips perf_tests which needs gflags;
# CUDA_ARCHITECTURES list must include your GPU: 80=A100, 90=H100)
cmake -B build . \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DPython_EXECUTABLE=$(which python) \
  -DSWIG_EXECUTABLE=$(which swig)

# Build & install
make -C build -j$(nproc) faiss swigfaiss
cd build/faiss/python && pip install .
```

Verify:

```bash
python -c "import faiss; print(faiss.__file__, faiss.__version__, hasattr(faiss,'GpuMultipleClonerOptions'))"
# Expected: .../site-packages/faiss/__init__.py 1.9.0 True
```

Notes:
- If you only have A100s, drop `90` from `CMAKE_CUDA_ARCHITECTURES`. For Blackwell (B100/B200) add `100`.
- `swig=4.2.*` is required; the v1.9.0 swig file does not compile with system `swig 3.x` (`SWIGTYPE_p_unsigned_long_long was not declared`) nor with `swig 4.4.x`.
- If a system `swig` is on `PATH`, force the conda one with `-DSWIG_EXECUTABLE=$(which swig)` and / or `export PATH=$CONDA_PREFIX/bin:$PATH` before re-running cmake.

#### CPU fallback

If you cannot get GPU faiss working (e.g. unsupported GPU arch), CPU faiss + GPU encoder is plenty fast for eval:

```bash
pip uninstall -y faiss faiss-gpu faiss-gpu-cu12 2>/dev/null
pip install faiss-cpu
# then drop --faiss_gpu from the retrieval_server.py launch command
```

### Step 3: Download Index and Corpus

**Note:** The local retrieval files are large. You'll need approximately **60-70 GB** for download and **132 GB** after extraction. Make sure you have sufficient disk space.

```bash
# Set your save path
save_path=/root/Index

# Download the index and corpus files
python /root/slime/examples/search-r1/local_dense_retriever/download.py --save_path $save_path

# Combine split index files
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress the corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 4: Start Local Retrieval Server

```bash
# If you encounter "conda not found" error, run:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda init
# source ~/.bashrc

# Activate retriever environment
conda activate retriever

# Make the env's libstdc++ visible (otherwise torch's bundled libnccl
# may complain: `libstdc++.so.6: version CXXABI_1.3.15' not found`).
# The cublas dir is only needed if you used the pip faiss-gpu-cu12 wheel.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set paths
save_path=/root/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Start the retrieval server
python /root/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

**Important Notes:**
- First startup will download the model and load the index, which may take a few minutes
- Normal startup time (excluding downloads): 1-2 minutes
- GPU memory usage per GPU: approximately 5-7 GB
- The local search engine's Python process will not terminate when the shell closes
- To restart the server: `lsof -i :8000` (`ss -ltnp 'sport = :8000'
ss -ltnp 'sport = :8001'
`) to find the PID, then kill it and restart

### Step 5: Start Training

Make sure you're **NOT** in the retriever conda environment. If you are, run `conda deactivate`.

```bash
cd /root/slime

# Set your wandb key (optional)
export WANDB_KEY="your_wandb_key_here"

# If ray process is stuck, try:
# rm -rf /root/.cache
# rm -rf /root/.*

# Run the training script
bash /root/slime/examples/search-r1/run_qwen2.5_3B.sh
```

### Troubleshooting

**Ray process stuck:**
```bash
rm -rf /root/.cache
# If still stuck:
rm -rf /root/.*
```

**Conda environment issues:**
- Make sure you deactivate the retriever environment before running training
- Verify you're using the base Python environment for training

**Retrieval server not responding:**
- Check if the server is running: `lsof -i :8000`
- Verify GPU availability: `nvidia-smi`
- Check logs for any error messages
