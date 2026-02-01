import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from session import Session, VLLM_BASE_URL, MODEL, API_KEY

VLLM_HEALTH_URL = VLLM_BASE_URL.rsplit("/v1", 1)[0] + "/health"

sessions: dict[str, Session] = {}


# Request/Response models — everything is OpenAI content blocks
class ContextRequest(BaseModel):
    session_id: str | None = None
    blocks: list[dict[str, Any]]  # OpenAI content blocks


class ActionRequest(BaseModel):
    session_id: str
    blocks: list[dict[str, Any]]  # OpenAI content blocks for one action


class PredictRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    ok: bool = True


class PredictResponse(BaseModel):
    session_id: str
    suggestion: str
    source: str | None
    bbox: list[int] | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup vLLM on startup."""
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
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Check API and vLLM health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(VLLM_HEALTH_URL)
            vllm_ok = r.status_code == 200
    except Exception:
        vllm_ok = False
    return {"ok": vllm_ok, "api": True, "vllm": vllm_ok}


@app.post("/context", response_model=SessionResponse)
async def set_context(req: ContextRequest):
    """Set user context (documents, instructions). Creates session if needed."""
    sid = req.session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = Session(session_id=sid)
    sessions[sid].set_context(req.blocks)
    return SessionResponse(session_id=sid)


@app.post("/action", response_model=SessionResponse)
async def add_action(req: ActionRequest):
    """Add action blocks. Call on each user action to warm KV cache."""
    if req.session_id not in sessions:
        sessions[req.session_id] = Session(session_id=req.session_id)
    sessions[req.session_id].add_action(req.blocks)
    return SessionResponse(session_id=req.session_id)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Get prediction. Call when user presses Cmd+E."""
    if req.session_id not in sessions:
        sessions[req.session_id] = Session(session_id=req.session_id)
    result = sessions[req.session_id].predict()
    return PredictResponse(
        session_id=req.session_id,
        suggestion=result["suggestion"],
        source=result["source"],
        bbox=result["bbox"],
    )
