#!/bin/bash

# If you encounter "conda not found" error, run:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda init
# source ~/.bashrc

set -ex

# Make `conda activate` available inside this non-interactive shell.
source /opt/conda/etc/profile.d/conda.sh

# Activate retriever environment
conda activate /tmp/user/.conda/envs/retriever
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

# Set paths
save_path=/mount/coreai-genai-pvc/baliao/experiments/agentic_opd/00_single_env/search_r1/data
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/mount/coreai-genai-pvc/baliao/experiments/PLLMs/intfloat/e5-base-v2

# Start 8 retrieval server processes, one per GPU, each on a different port (8000-8007)
for gpu_id in {0..7}; do
  port=$((8000 + gpu_id))
  echo "Starting retrieval server on GPU $gpu_id, port $port..."
  CUDA_VISIBLE_DEVICES=$gpu_id nohup python local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port $port > retrieval_server_gpu${gpu_id}.log 2>&1 &
done

echo "All retrieval servers started. Use a load balancer (e.g., nginx) to distribute requests to ports 8000-8007."