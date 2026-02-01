import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import openai
from fastapi import FastAPI
from pydantic import BaseModel

# Config
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
API_KEY = os.getenv("API_KEY", "EMPTY")
MAX_PREDICT_TOKENS = int(os.getenv("MAX_PREDICT_TOKENS", "100"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
VLLM_HEALTH_URL = VLLM_BASE_URL.rsplit("/v1", 1)[0] + "/health"

SYSTEM_PROMPT = """You are a desktop autocomplete assistant.

## When you're called
The user has selected an empty text field and pressed Cmd+E to request a suggestion.

## Your task
Generate a suggestion for what to enter in the text field based on:
1. The user's context (documents, instructions, personal info) — provided as text and images
2. Recent actions — what the user has been doing on screen

## Context format
Context contains interlaced text and images. Images are labeled like [c1], [c2], etc.
Example:
- Text: "My driver's license:"
- [c1] followed by an image of the license

## Response format
Return raw JSON only (no markdown, no ```json fences):
{"suggestion": "text to fill", "source": "c1", "bbox": [x1, y1, x2, y2]}

- `suggestion`: The text to enter in the field. Keep it concise and relevant.
- `source`: If the info comes from an image, return its label (e.g., "c1"). Otherwise null.
- `bbox`: If source is an image, draw a bounding box around the relevant info. Coordinates are normalized 0-1000 (top-left origin). Format: [x1, y1, x2, y2]. Otherwise null.

## When to skip
If there's no clear evidence in the context for what to fill, return:
{"suggestion": "<skip>", "source": null, "bbox": null}

Only suggest information you can find in the provided context. Do not hallucinate."""

# Session state: {session_id: {"context": [...], "actions": [...]}}
sessions: dict[str, dict] = {}

# OpenAI client
client = openai.OpenAI(api_key=API_KEY, base_url=VLLM_BASE_URL)


# Request/Response models
class ContextRequest(BaseModel):
    session_id: str | None = None
    blocks: list[dict[str, Any]]


class ActionRequest(BaseModel):
    session_id: str
    blocks: list[dict[str, Any]]


class PredictRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    ok: bool = True


class PredictResponse(BaseModel):
    session_id: str
    raw: str


def get_session(sid: str) -> dict:
    if sid not in sessions:
        sessions[sid] = {"context": [], "actions": []}
    return sessions[sid]


def call_model(content: list[dict], max_tokens: int = MAX_PREDICT_TOKENS) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


@asynccontextmanager
async def lifespan(app: FastAPI):
    call_model([{"type": "text", "text": "hi"}], max_tokens=1)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(VLLM_HEALTH_URL)
            vllm_ok = r.status_code == 200
    except Exception:
        vllm_ok = False
    return {"ok": vllm_ok, "api": True, "vllm": vllm_ok}


@app.post("/context", response_model=SessionResponse)
async def set_context(req: ContextRequest):
    sid = req.session_id or str(uuid.uuid4())
    session = get_session(sid)
    session["context"] = req.blocks
    call_model(session["context"], max_tokens=1)
    return SessionResponse(session_id=sid)


@app.post("/action", response_model=SessionResponse)
async def add_action(req: ActionRequest):
    session = get_session(req.session_id)
    session["actions"].extend(req.blocks)
    call_model(session["context"] + session["actions"], max_tokens=1)
    return SessionResponse(session_id=req.session_id)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    session = get_session(req.session_id)
    raw = call_model(session["context"] + session["actions"], max_tokens=100)
    return PredictResponse(session_id=req.session_id, raw=raw)
