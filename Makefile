# Config (override via env or command line)
export MODEL           ?= Qwen/Qwen3-VL-32B-Instruct-FP8
export DTYPE           ?= auto
export TENSOR_PARALLEL ?= 1
export GPUS            ?= all
export MAX_MODEL_LEN   ?= 100000
export MAX_NUM_SEQS    ?= 128
export GPU_MEMORY_UTILIZATION ?= 0.95
export HF_CACHE        ?= $(HOME)/.cache/huggingface
export API_PORT        ?= 443
export MAX_PREDICT_TOKENS ?= 100
export MAX_COMPACT_TOKENS ?= 200
export MAX_ACTIONS     ?= 0
export TEMPERATURE     ?= 0.0

IMAGE_NAME ?= desktop-autocomplete
CONTAINER  ?= desktop-autocomplete

.PHONY: build run stop restart logs shell clean

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .
	@echo "✅ Image built: $(IMAGE_NAME)"

run: ## Run the container
	docker run -d \
		--name $(CONTAINER) \
		--gpus $(GPUS) \
		--ipc=host \
		--shm-size=16g \
		-p $(API_PORT):8080 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		-e MODEL=$(MODEL) \
		-e DTYPE=$(DTYPE) \
		-e TENSOR_PARALLEL=$(TENSOR_PARALLEL) \
		-e MAX_MODEL_LEN=$(MAX_MODEL_LEN) \
		-e MAX_NUM_SEQS=$(MAX_NUM_SEQS) \
		-e GPU_MEMORY_UTILIZATION=$(GPU_MEMORY_UTILIZATION) \
		-e MAX_PREDICT_TOKENS=$(MAX_PREDICT_TOKENS) \
		-e MAX_COMPACT_TOKENS=$(MAX_COMPACT_TOKENS) \
		-e MAX_ACTIONS=$(MAX_ACTIONS) \
		-e TEMPERATURE=$(TEMPERATURE) \
		-e VLLM_DEEP_GEMM_ENABLE=0 \
		-v $(CURDIR)/app.py:/app/app.py:ro \
		$(IMAGE_NAME)
	@echo "✅ Container started. Run 'make logs' to watch."

stop: ## Stop the container
	docker stop $(CONTAINER) && docker rm $(CONTAINER)

restart: stop run ## Restart the container

logs: ## Tail container logs
	docker logs -f $(CONTAINER)

shell: ## Shell into the container
	docker exec -it $(CONTAINER) /bin/bash

clean: stop ## Stop and remove image
	docker rmi $(IMAGE_NAME)
