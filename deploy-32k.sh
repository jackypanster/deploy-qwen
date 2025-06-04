#!/bin/bash

docker run -d \
  --runtime=nvidia \
  --gpus=all \
  --name coder \
  -v /home/llm/model/qwen/qwen3-30b-a3b:/qwen/qwen3-30b-a3b \
  -p 8000:8000 \
  --cpuset-cpus 0-55 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart always \
  --ipc=host \
  vllm/vllm-openai:v0.8.5 \
  --model /qwen/qwen3-30b-a3b \
  --served-model-name coder \
  --tensor-parallel-size 4 \
  --dtype half \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.93 \
  --block-size 32 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --tokenizer-pool-size 56 \
  --disable-custom-all-reduce
