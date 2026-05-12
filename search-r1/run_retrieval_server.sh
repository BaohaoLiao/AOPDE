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

# Start 2 retrieval server processes, each using 4 GPUs (0-3 and 4-7), sharding the index across 4 GPUs per server
for group in 0 1; do
  if [ $group -eq 0 ]; then
    gpus="0,1,2,3"
    port=8000
  else
    gpus="4,5,6,7"
    port=8001
  fi
  echo "Starting retrieval server on GPUs $gpus, port $port..."
  CUDA_VISIBLE_DEVICES=$gpus nohup python local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu \
    --port $port > retrieval_server_gpus_${gpus//,/}.log 2>&1 &
done

echo "Both retrieval servers started. Use a load balancer (e.g., nginx) to distribute requests to ports 8000 and 8001."