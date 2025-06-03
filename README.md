# Qwen3-30B Deployment Guide

This repository contains scripts and configurations for deploying the Qwen3-30B model using vLLM with OpenAI API compatibility.

## System Requirements

- **Hardware**: 
  - 4x NVIDIA GPUs (22GB VRAM each, 88GB total)
  - 512GB RAM
  - 2TB SSD storage
  - 56 CPU cores
- **Software**:
  - Ubuntu 24.04
  - NVIDIA Driver 550.144.03
  - CUDA 12.4
  - Docker with NVIDIA runtime

## Model Information

- **Model**: Qwen3-30B-A3B
- **Context Length**: 32K tokens (configurable)
- **Deployment Method**: vLLM with tensor parallelism across 4 GPUs

## Deployment Script

The `deploy.sh` script contains the Docker command to deploy the model:

```bash
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
```

## Key Parameters Explained

- `--tensor-parallel-size 4`: Distributes model across all 4 GPUs
- `--dtype half`: Uses FP16 precision to reduce memory usage
- `--max-model-len 32768`: Sets maximum context length to 32K tokens
- `--max-num-batched-tokens 4096`: Batch size for token processing
- `--gpu-memory-utilization 0.93`: Uses 93% of available GPU memory
- `--swap-space 16`: Allocates 16GB of CPU memory for GPU memory swapping
- `--tokenizer-pool-size 56`: Matches CPU count for optimal tokenization

## Usage

After deployment, the model exposes an OpenAI-compatible API at:

```
http://localhost:8000
```

### Available Endpoints

- `/v1/chat/completions` - For chat completion requests
- `/v1/completions` - For text completion requests
- `/v1/models` - To list available models

### Example API Call

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## Advanced Configuration

The Qwen3-30B model natively supports up to 32K tokens. For even longer contexts (up to 131K tokens), YaRN scaling techniques can be implemented as described in the [model documentation](https://huggingface.co/Qwen/Qwen3-30B-A3B).

## Monitoring

Check container logs for deployment status and errors:

```bash
docker logs -f coder
```

## Resource Usage

- Each GPU uses approximately 20.15GB of memory with the current configuration
- Model weights: 14.25GB per GPU
- KV cache: ~5.47GB per GPU
