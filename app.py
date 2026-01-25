import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from session import Session, VLLM_BASE_URL, MODEL, API_KEY

VLLM_HEALTH_URL = VLLM_BASE_URL.rsplit("/v1", 1)[0] + "/health"

sessions: dict[str, Session] = {}


class PredictRequest(BaseModel):
    session_id: str | None = None
    action: dict[str, Any]  # Flattened action object from FE


class PredictResponse(BaseModel):
    session_id: str
    prediction: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup vLLM on startup. Compose ensures vLLM is healthy first."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello?"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    print("✅ Warmup complete")
    yield  # App runs here
    # Shutdown logic (if any) goes after yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Returns ok=True only if both API and vLLM are healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(VLLM_HEALTH_URL)
            vllm_ok = r.status_code == 200
    except Exception:
        vllm_ok = False

    return {"ok": vllm_ok, "api": True, "vllm": vllm_ok}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = Session(session_id=sid)
    session = sessions[sid]
    session.add_action(req.action)
    prediction = session.inference()
    return PredictResponse(session_id=session.session_id, prediction=prediction)
