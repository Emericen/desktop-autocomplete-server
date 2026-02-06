# Inference Server

Single-container FastAPI server with vLLM's offline `LLM` class for vision-language inference with session management.

## Quick Start

```bash
# Build image, start container, tail logs
make build && make run && make logs
```

## Architecture

The server embeds vLLM's offline `LLM` class directly in the FastAPI process — no separate vLLM server or docker compose needed. The model loads on startup via the FastAPI lifespan hook.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (model loaded status) |
| `/clipboard` | POST | Set clipboard content for a session |
| `/action` | POST | Add user actions to a session |
| `/predict` | POST | Generate autocomplete suggestion |
| `/compact` | POST | Summarize actions to reduce token count |
| `/session/{id}` | GET | Debug: view full session state |

## Commands

```bash
make build     # Build the Docker image
make run       # Start the container (GPU)
make stop      # Stop and remove container
make restart   # Restart container
make logs      # Tail container logs
make shell     # Shell into container
make clean     # Stop and remove image
```

## Configuration

All configurable via environment variables or Makefile overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `TENSOR_PARALLEL` | `1` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | `100000` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory fraction for KV cache |
| `API_PORT` | `443` | Host port mapped to container |
| `TEMPERATURE` | `0.0` | Sampling temperature |
