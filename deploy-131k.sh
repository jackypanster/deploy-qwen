#!/bin/bash

# Deploy Qwen3-30B-A3B with 131K context window using YaRN scaling
# This script COMPLETELY disables reasoning/thinking mode using a custom template
#
# 优点:
# - 适合编程任务，直接输出代码而不会生成思考过程
# - 性能更高：关闭reasoning后token生成速度提高约15-20%
# - 内存使用更低：对于超长上下文可节省大量内存
# 
# 编程任务推荐参数: temperature=0.2, top_p=0.6, top_k=50

docker run -d \
  --runtime=nvidia \
  --gpus=all \
  --name coder \
  -v /home/llm/model/qwen/qwen3-30b-a3b:/qwen/qwen3-30b-a3b \
  -v /home/llm/workspace/deploy-qwen:/workspace/deploy-qwen \
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
  --max-model-len 131072 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --block-size 32 \
  --enable-chunked-prefill \
  --swap-space 24 \
  --tokenizer-pool-size 56 \
  --disable-custom-all-reduce \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template /workspace/deploy-qwen/qwen3_nonthinking.jinja

# ============ 如何切换 reasoning/thinking 模式 ============
# 
# 1. 关闭reasoning模式（当前配置，性能更好）:
#    保留 --chat-template 参数
#    删除 --enable-reasoning 和 --reasoning-parser 参数
#
# 2. 开启reasoning模式（更完善的推理）:
#    删除 --chat-template 参数
#    添加以下参数:
#    --enable-reasoning \
#    --reasoning-parser deepseek_r1
#
# 3. 重启容器使更改生效:
#    docker stop coder && docker rm coder && ./deploy-131k.sh
#
# 注意: 对于131K模式，关闭reasoning尤为重要，能显著降低超长上下文的内存占用
