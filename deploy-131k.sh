#!/bin/bash

# Deploy Qwen3-30B-A3B with 131K context window using YaRN scaling
# This script uses a specialized template that enhances programming capabilities while disabling reasoning
#
# 优点:
# - 编程任务优化：强化了代码质量，含默认编程系统提示
# - 性能更高：关闭reasoning后token生成速度提高约15-20%
# - 代码与超长文本处理：结合131K上下文能力处理大型代码库
# - 内存效率极高：对于超长上下文能显著降低内存占用
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
  --chat-template /workspace/deploy-qwen/qwen3_programming.jinja

# ============ 模板选择和模式切换 ============
# 
# 1. 编程优化模式（当前配置，默认全局编程系统提示）:
#    --chat-template /workspace/deploy-qwen/qwen3_programming.jinja
#
# 2. 纯禁用思考模式（无默认系统提示）:
#    --chat-template /workspace/deploy-qwen/qwen3_nonthinking.jinja
#
# 3. 开启reasoning模式（更完善的推理）:
#    删除 --chat-template 参数
#    添加以下参数:
#    --enable-reasoning \
#    --reasoning-parser deepseek_r1
#
# 4. 重启容器使更改生效:
#    docker stop coder && docker rm coder && ./deploy-131k.sh
#
# 注意: 对于131K模式，使用编程优化模板特别重要，可充分利用超长上下文进行复杂代码库的分析处理
