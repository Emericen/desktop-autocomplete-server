# Config (override via env or command line)
export MODEL      ?= Qwen/Qwen3-VL-32B-Instruct-FP8
export DTYPE      ?= auto
export TENSOR_PARALLEL ?= 1
export GPUS       ?= all
export MAX_MODEL_LEN ?= 100000
export MAX_IMAGE_COUNT ?= 100
export TOOL_CALL_PARSER ?= hermes
export HF_CACHE   ?= $(HOME)/.cache/huggingface
# Internal: vLLM ↔ API communication
export VLLM_PORT  ?= 8000
# External: exposed to host
export API_PORT   ?= 443
export MAX_PREDICT_TOKENS ?= 20
export MAX_COMPACT_TOKENS ?= 200
export TEMPERATURE ?= 0.0

# Performance & Memory
export GPU_MEMORY_UTILIZATION ?= 0.95
export MAX_NUM_SEQS ?= 128
export PREFIX_CACHING_HASH_ALGO ?= xxhash

.PHONY: up down restart logs logs-vllm logs-api shell warmup build clean

up: ## Start all services
	docker compose up -d
	@echo "✅ Services starting. Run 'make logs' to watch."

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

logs: ## Tail all logs
	docker compose logs -f

logs-vllm: ## Tail vLLM logs only
	docker compose logs -f vllm

logs-api: ## Tail API logs only
	docker compose logs -f api

shell: ## Shell into vLLM container
	docker compose exec vllm /bin/bash

build: ## Rebuild API container
	docker compose build api

restart-api: ## Restart only the API container (keeps vLLM running)
	docker compose restart api

warmup: ## Preload vision processor
	@echo "Sending warmup request..."
	@curl -s -X POST http://localhost:$(API_PORT)/health -o /dev/null -w "API ready (%{http_code})\n"

clean: down ## Stop and remove images
	docker compose down --rmi all --volumes
