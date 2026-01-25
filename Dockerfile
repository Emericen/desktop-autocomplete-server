FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY *.py .

ENV API_PORT=8080
EXPOSE ${API_PORT}

CMD uvicorn app:app --host 0.0.0.0 --port ${API_PORT}
