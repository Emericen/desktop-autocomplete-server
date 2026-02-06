FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Install FastAPI + uvicorn on top of vLLM base image
RUN pip install --no-cache-dir fastapi==0.128.0 uvicorn==0.40.0

COPY app.py .

ENV API_PORT=8080
EXPOSE ${API_PORT}

# Override the vLLM entrypoint so we run our own FastAPI app
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
