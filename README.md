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

## Deployment Scripts

This repository now provides two deployment scripts tailored for different context length requirements. Both scripts use the container name `coder` and served model name `coder` for seamless switching without client-side changes.

1.  **`deploy-131k.sh`**: Deploys the Qwen3-30B model with an extended context length of **131,072 tokens** using YaRN scaling. Use this for tasks requiring very long context processing.
2.  **`deploy-32k.sh`**: Deploys the Qwen3-30B model with its native context length of **32,768 tokens**. Use this for general tasks, offering higher concurrency and potentially faster responses for shorter inputs compared to the 131K version.

To use them, first ensure they are executable:
```bash
chmod +x deploy-131k.sh
chmod +x deploy-32k.sh
```
Then run the desired script:
```bash
./deploy-131k.sh # For 131K context
# OR
./deploy-32k.sh  # For 32K context
```

**Note**: Only one instance can run at a time due to shared GPU resources and port mapping. Stop the current `coder` container before starting a new one with a different configuration:
```bash
docker stop coder && docker rm coder
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

### Model and vLLM Parameters

The following table highlights key vLLM parameters and their settings in the two deployment scripts:

| Parameter                       | `deploy-131k.sh` Value                                       | `deploy-32k.sh` Value | Description                                                                 |
|---------------------------------|--------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------|
| `--tensor-parallel-size`        | `4`                                                          | `4`                   | Distributes model across all 4 GPUs.                                        |
| `--dtype`                       | `half`                                                       | `half`                | Uses FP16 precision to reduce memory usage.                                 |
| `--max-model-len`               | `131072`                                                     | `32768`               | Maximum context length in tokens.                                           |
| `--block-size`                  | `32`                                                         | `32`                  | Sets KV cache block size.                                                   |
| `--enable-chunked-prefill`      | `true` (implied by presence)                                 | `true` (implied)      | Enables processing long sequences in chunks, crucial for long contexts.     |
| `--served-model-name`           | `coder`                                                      | `coder`               | Names the model for API requests.                                           |
| `--max-num-batched-tokens`      | `2048`                                                       | `4096`                | Max tokens in a batch. Reduced for 131K due to higher memory per token.   |
| `--gpu-memory-utilization`      | `0.92`                                                       | `0.93`                | GPU memory utilization. Adjusted for stability with different contexts.   |
| `--swap-space`                  | `24` (GB)                                                    | `16` (GB)             | CPU memory for GPU swapping. Increased for 131K.                          |
| `--tokenizer-pool-size`         | `56`                                                         | `56`                  | Matches CPU count for optimal tokenization.                                 |
| `--disable-custom-all-reduce`   | `true` (implied by presence)                                 | `true` (implied)      | Disables custom all-reduce for better compatibility.                        |
| `--rope-scaling`                | `'{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'` | Not present           | Enables YaRN scaling for 131K context. Only in `deploy-131k.sh`.          |

### YaRN Scaling for Extended Context (Specific to `deploy-131k.sh`)
- The `--rope-scaling` parameter is exclusively used in `deploy-131k.sh`.
  - Enables YaRN scaling to extend position embeddings.
  - Uses factor 4.0 to scale from native 32K to 131K tokens.
  - Specifies original position embedding size (32768).

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

## YaRN Scaling Implementation (Specific to `deploy-131k.sh`)

The `deploy-131k.sh` script uses YaRN (Yet another RoPE extension) scaling to extend the model's context window:

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

## Reasoning Mode: Performance Implications & How to Toggle

Qwen3-30B-A3B features a "thinking/reasoning mode" that can be toggled on or off. The deployment scripts have been updated to **completely disable reasoning mode** using a custom template for better performance, especially for programming tasks.

### What is Reasoning Mode?

Reasoning mode (also called thinking mode) is a capability where the model explicitly generates intermediate reasoning steps (enclosed in `<think>...</think>` tags) before producing a final answer. While this can improve reasoning quality for complex tasks, it adds significant overhead for straightforward programming tasks.

### Performance Benefits of Disabled Reasoning

Disabling reasoning mode provides several important benefits:

1. **Faster Response Generation**: 
   - Approximately 15-20% increase in token generation speed
   - More direct answers without verbose thinking steps

2. **Reduced Memory Usage**:
   - No need to allocate token space for reasoning content
   - Especially important for the 131K context model where memory efficiency is critical

3. **More Concise Outputs**:
   - Directly generates code without intermediate reasoning steps
   - Better suited for programming tasks that need precise, focused outputs

### How to Toggle Reasoning Mode

#### Current Implementation: Completely Disabled

Both deployment scripts use a **custom chat template** that completely disables reasoning mode:

```bash
--chat-template /workspace/deploy-qwen/qwen3_nonthinking.jinja
```

This approach is more robust than API-level parameters because it ensures the model cannot generate reasoning content even if instructed to at the API level.

#### Important Volume Mount Update

The deployment scripts now include an additional volume mount to make the template file accessible inside the Docker container:

```bash
-v /home/llm/workspace/deploy-qwen:/workspace/deploy-qwen
```

#### Step-by-Step Instructions for Managing Reasoning Mode

1. **Keep Reasoning Disabled (Default, Recommended for Programming)**
   - Keep the `--chat-template` parameter in the deployment scripts
   - Delete any `--enable-reasoning` and `--reasoning-parser` parameters

2. **Enable Reasoning (If Complex Reasoning is Needed)**
   - Edit the deployment script
   - Remove the `--chat-template` parameter
   - Add these parameters:
     ```bash
     --enable-reasoning \
     --reasoning-parser deepseek_r1
     ```

3. **Apply Changes**
   - Restart the Docker container:
     ```bash
     docker stop coder && docker rm coder
     ./deploy-32k.sh  # or ./deploy-131k.sh
     ```

#### Client-Side Control

When making API requests, you can control thinking mode through the `enable_thinking` parameter:

```bash
# Disable thinking mode (for programming tasks)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "enable_thinking": false,
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list of integers."}
    ],
    "temperature": 0.2,
    "top_p": 0.6,
    "top_k": 50
  }'
```

```bash
# Enable thinking mode (default behavior)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "enable_thinking": true,
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list of integers."}
    ],
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

### Optimized Parameters for Programming Tasks

For programming-related tasks, it's recommended to:

1. **Disable thinking mode** by setting `"enable_thinking": false` in your API requests
2. **Use lower temperature settings** for more deterministic and precise code generation

#### Recommended Parameters for Programming

| Parameter     | Value | Explanation |
|---------------|-------|-------------|
| `temperature` | 0.2   | Lower temperature for more deterministic code generation |
| `top_p`       | 0.6   | More focused token selection for code precision |
| `top_k`       | 50    | Limits vocabulary to most relevant tokens for code |
| `min_p`       | 0.0   | Standard setting for sampling diversity |

#### Example API Request for Programming Tasks

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coder",
    "enable_thinking": false,
    "messages": [
      {"role": "user", "content": "Write a Python function to implement binary search."}
    ],
    "temperature": 0.2,
    "top_p": 0.6,
    "top_k": 50
  }'
```

This configuration helps produce more precise, syntactically correct, and concise code for programming tasks while avoiding unnecessary reasoning steps.

## Monitoring

Check container logs for deployment status and errors:

```bash
docker logs -f coder
```

### Key Log Indicators
When reviewing logs (`docker logs -f coder`), look for these important signals depending on the script used:

**Common to both deployments:**
- **Memory Allocation** (example values, will vary slightly based on `gpu-memory-utilization`):
  ```
  INFO [worker.py:287] the current vLLM instance can use total_gpu_memory (21.66GiB) x gpu_memory_utilization (0.9X) = XX.XXGiB
  ```
- **API Server Started**:
  ```
  INFO [api_server.py:1090] Starting vLLM API server on http://0.0.0.0:8000
  ```

**Specific to `deploy-131k.sh`:**
- **YaRN Configuration Applied**:
  ```
  INFO [config.py:456] Overriding HF config with {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}}
  ```
- **Context Length Confirmation (131K)**:
  ```
  INFO [llm_engine.py:240] ... max_seq_len=131072 ...
  ```
- **Concurrency Capability (131K)**:
  ```
  INFO [executor_base.py:117] Maximum concurrency for 131072 tokens per request: 1.21x
  ```

**Specific to `deploy-32k.sh`:**
- **Context Length Confirmation (32K)**:
  ```
  INFO [llm_engine.py:240] ... max_seq_len=32768 ...
  ```
- **Concurrency Capability (32K)** (Example, actual value may vary):
  ```
  INFO [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 5.73x 
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
