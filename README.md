# Qwen3-30B Deployment Guide

This repository contains scripts and configurations for deploying the Qwen3-30B model using vLLM with OpenAI API compatibility, featuring extended context length capabilities up to 131K tokens.

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
- **Context Length Options**:
  - 32K tokens (natively supported)
  - 131K tokens (with YaRN scaling)
- **Deployment Method**: vLLM with tensor parallelism across 4 GPUs

## Deployment Script

The `deploy.sh` script contains the Docker command to deploy the model with 131K context length:

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
  --max-model-len 131072 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --block-size 32 \
  --enable-chunked-prefill \
  --swap-space 24 \
  --tokenizer-pool-size 56 \
  --disable-custom-all-reduce \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

## Key Parameters Explained

### Docker Options
- `--runtime=nvidia`: Uses NVIDIA container runtime
- `--gpus=all`: Enables access to all GPUs
- `--name coder`: Names the container for easy reference
- `-v /home/llm/model/qwen/qwen3-30b-a3b:/qwen/qwen3-30b-a3b`: Mounts model directory
- `-p 8000:8000`: Exposes API on port 8000
- `--cpuset-cpus 0-55`: Pins container to specific CPU cores (0-55)
- `--ulimit memlock=-1`: Removes memory lock limits
- `--ulimit stack=67108864`: Sets stack size limit to 64MB
- `--ipc=host`: Uses host IPC namespace for better shared memory

### Model Configuration
- `--tensor-parallel-size 4`: Distributes model across all 4 GPUs
- `--dtype half`: Uses FP16 precision to reduce memory usage
- `--max-model-len 131072`: Sets maximum context length to 131K tokens
- `--block-size 32`: Sets KV cache block size
- `--enable-chunked-prefill`: Enables processing long sequences in chunks
- `--served-model-name coder`: Names the model for API requests

### Memory and Performance Parameters
- `--max-num-batched-tokens 2048`: Reduced batch size for processing tokens with longer contexts
- `--gpu-memory-utilization 0.92`: Uses 92% of available GPU memory (optimized based on actual usage)
- `--swap-space 24`: Allocates 24GB of CPU memory for GPU memory swapping (increased from 16GB)
- `--tokenizer-pool-size 56`: Matches CPU count for optimal tokenization
- `--disable-custom-all-reduce`: Disables custom all-reduce for better compatibility

### YaRN Scaling for Extended Context
- `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`: 
  - Enables YaRN scaling to extend position embeddings
  - Uses factor 4.0 to scale from native 32K to 131K tokens
  - Specifies original position embedding size (32768)

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

## YaRN Scaling Implementation

The deployment now uses YaRN (Yet another RoPE extension) scaling to extend the model's context window:

- **Native Context**: 32,768 tokens
- **Extended Context**: 131,072 tokens (4x scaling)

### What is YaRN?
YaRN is a positional embedding scaling technique that allows models to process sequences much longer than they were trained on. It applies a scaling factor to the RoPE (Rotary Position Embedding) parameters while maintaining performance on shorter sequences.

### Implementation Details
- **Scaling Factor**: 4.0 (extends from 32K to 131K)
- **Original Max Position**: 32,768 tokens
- **RoPE Type**: yarn

### Performance Implications
As revealed in the logs, extending context length significantly impacts concurrency:
- **32K Context**: 5.73x maximum concurrency
- **131K Context**: 1.21x maximum concurrency

For details, see the official [Qwen3-30B-A3B model documentation](https://huggingface.co/Qwen/Qwen3-30B-A3B).

## Monitoring

Check container logs for deployment status and errors:

```bash
docker logs -f coder
```

### Key Log Indicators
When reviewing logs, look for these important signals:

1. **YaRN Configuration Applied**:
   ```
   INFO [config.py:456] Overriding HF config with {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}}
   ```

2. **Context Length Confirmation**:
   ```
   INFO [llm_engine.py:240] ... max_seq_len=131072 ...
   ```

3. **Memory Allocation**:
   ```
   INFO [worker.py:287] the current vLLM instance can use total_gpu_memory (21.66GiB) x gpu_memory_utilization (0.90) = 19.50GiB
   ```

4. **Concurrency Capability**:
   ```
   INFO [executor_base.py:117] Maximum concurrency for 131072 tokens per request: 1.21x
   ```

5. **API Server Started**:
   ```
   INFO [api_server.py:1090] Starting vLLM API server on http://0.0.0.0:8000
   ```

### Warning Analysis
You may see these warnings which are safe to ignore:

- `Using default MoE config`: Normal for non-MoE model configurations
- `Cannot use FlashAttention-2 backend for Volta and Turing GPUs`: Expected for older GPU architectures
- `Using XFormers backend`: Shows proper fallback attention mechanism

## Resource Usage

- **GPU Memory**: ~19.93GB per GPU (92% of 21.66GB)
- **Model Weights**: 14.27GB per GPU 
- **KV Cache**: ~4.80GB per GPU (reduced from 32K configuration)
- **Loading Time**: ~20 seconds for model weights
- **Graph Capture**: ~34 seconds for CUDA graph capturing
