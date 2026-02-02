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
MAX_COMPACT_TOKENS = int(os.getenv("MAX_COMPACT_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
VLLM_HEALTH_URL = VLLM_BASE_URL.rsplit("/v1", 1)[0] + "/health"

SYSTEM_PROMPT = """You are a desktop autocomplete assistant.

## When you're called
The user has selected an empty text field and pressed Cmd+E to request a suggestion.

## Your task
Generate a suggestion for what to enter in the text field based on:
1. The user's clipboard (documents, instructions, personal info) — provided as text and images
2. Recent actions — what the user has been doing on screen

## Clipboard format
Clipboard contains interlaced text and images. Images are labeled like [c1], [c2], etc.
Example:
- Text: "My driver's license:"
- [c1] followed by an image of the license

## Response format
Return raw JSON only (no markdown, no ```json fences):
{"suggestion": "text to fill", "source": "c1", "bbox": [x1, y1, x2, y2]}

- `suggestion`: The text to enter in the field. Keep it concise and relevant.
- `source`: If the info comes from an image, return its label (e.g., "c1"). Otherwise null.
- `bbox`: If source is an image, draw a bounding box around the relevant info. Coordinates are normalized 0-1000 (top-left origin). Format: [x1, y1, x2, y2]. Otherwise null.

## CRITICAL: When to skip (DO NOT HALLUCINATE)
If the information the user needs is NOT directly visible in the clipboard or action history, you MUST return:
{"suggestion": null, "source": null, "bbox": null}

Skip when:
- The field asks for info not present in clipboard (e.g., phone number when only address is available)
- You're unsure what the field is asking for
- The info cannot be verified from the provided content

NEVER make up, guess, or infer information that isn't explicitly shown. It's better to return null than to suggest something incorrect. The user trusts you to only provide verified information."""

COMPACT_PROMPT = """Summarize the user's recent actions into a brief paragraph. Extract key information relevant to understanding what the user is doing and what they might need to fill in next. Be concise."""

# Session state: {session_id: {"clipboard": [...], "actions": [], "tokens": 0}}
sessions: dict[str, dict] = {}

# OpenAI client
client = openai.OpenAI(api_key=API_KEY, base_url=VLLM_BASE_URL)


# Request/Response models
class ClipboardRequest(BaseModel):
    session_id: str | None = None
    blocks: list[dict[str, Any]]


class ActionRequest(BaseModel):
    session_id: str
    blocks: list[dict[str, Any]]


class SessionRequest(BaseModel):
    session_id: str


class ActionResponse(BaseModel):
    session_id: str
    prompt_tokens: int
    ok: bool = True


class CompactResponse(BaseModel):
    session_id: str
    prompt_tokens: int
    summary: str


class PredictResponse(BaseModel):
    session_id: str
    prompt_tokens: int
    raw: str


class TestRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 50
    temperature: float = 0.0


def get_session(sid: str) -> dict:
    if sid not in sessions:
        sessions[sid] = {"clipboard": [], "actions": [], "tokens": 0}
    return sessions[sid]


def call_model(
    sid: str | None,
    content: list[dict],
    max_tokens: int = MAX_PREDICT_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, int]:
    """Call model and return (content, prompt_tokens)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    # Track tokens in session
    if sid and sid in sessions:
        sessions[sid]["tokens"] = prompt_tokens
    return response.choices[0].message.content, prompt_tokens


@asynccontextmanager
async def lifespan(app: FastAPI):
    call_model(None, [{"type": "text", "text": "hi"}], max_tokens=1)
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


@app.post("/test")
def test(req: TestRequest):
    """Raw passthrough to vLLM for testing."""
    response = client.chat.completions.create(
        model=req.model,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return response.model_dump()


@app.post("/clipboard", response_model=ActionResponse)
async def set_clipboard(req: ClipboardRequest):
    """Set user's clipboard (documents, personal info). Static, not compacted."""
    sid = req.session_id or str(uuid.uuid4())
    session = get_session(sid)
    session["clipboard"] = [{"type": "text", "text": "Clipboard:"}] + req.blocks
    _, prompt_tokens = call_model(sid, session["clipboard"], max_tokens=1)
    return ActionResponse(session_id=sid, prompt_tokens=prompt_tokens)


@app.post("/action", response_model=ActionResponse)
async def add_action(req: ActionRequest):
    """Add user actions. Returns token count so client knows when to compact."""
    session = get_session(req.session_id)
    session["actions"].extend(
        [{"type": "text", "text": "Recent Actions:"}] + req.blocks
    )
    content = session["clipboard"] + session["actions"]
    _, prompt_tokens = call_model(req.session_id, content, max_tokens=1)
    return ActionResponse(session_id=req.session_id, prompt_tokens=prompt_tokens)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: SessionRequest):
    """Generate autocomplete suggestion."""
    session = get_session(req.session_id)
    content = session["clipboard"] + session["actions"]
    raw, prompt_tokens = call_model(
        req.session_id, content, max_tokens=MAX_PREDICT_TOKENS
    )
    return PredictResponse(
        session_id=req.session_id, prompt_tokens=prompt_tokens, raw=raw
    )


@app.post("/compact", response_model=CompactResponse)
async def compact(req: SessionRequest):
    """Summarize actions into a short text, clear actions, return new token count."""
    session = get_session(req.session_id)

    # Build content: clipboard + actions + compact instruction
    content = (
        session["clipboard"]
        + session["actions"]
        + [{"type": "text", "text": f"\n\n{COMPACT_PROMPT}"}]
    )

    # Get summary
    summary, _ = call_model(
        req.session_id,
        content,
        max_tokens=MAX_COMPACT_TOKENS,
        system_prompt=COMPACT_PROMPT,
    )

    # Replace actions with summary
    session["actions"] = [
        {"type": "text", "text": f"...\n[Previous Actions Summary]\n{summary}"}
    ]

    # Get new token count
    new_content = session["clipboard"] + session["actions"]
    _, prompt_tokens = call_model(req.session_id, new_content, max_tokens=1)

    return CompactResponse(
        session_id=req.session_id, prompt_tokens=prompt_tokens, summary=summary
    )


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Debug endpoint: view full session state (clipboard + actions + tokens)."""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "prompt_tokens": session["tokens"],
        "clipboard_blocks": len(session["clipboard"]),
        "action_blocks": len(session["actions"]),
        "clipboard": session["clipboard"],
        "actions": session["actions"],
    }
