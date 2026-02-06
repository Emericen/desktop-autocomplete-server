FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Install dependencies on top of vLLM base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV API_PORT=8080
EXPOSE ${API_PORT}

# Override the vLLM entrypoint so we run our own FastAPI app
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
