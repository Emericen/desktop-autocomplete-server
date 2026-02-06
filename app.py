import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Config
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", "1"))
DTYPE = os.getenv("DTYPE", "auto")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "100000"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "128"))
MAX_PREDICT_TOKENS = int(os.getenv("MAX_PREDICT_TOKENS", "100"))
MAX_COMPACT_TOKENS = int(os.getenv("MAX_COMPACT_TOKENS", "300"))
MAX_ACTIONS = int(
    os.getenv("MAX_ACTIONS", "0")
)  # 0 = unlimited, N = sliding window of last N action pairs
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

SYSTEM_PROMPT = """You are a desktop autocomplete assistant.

The user clicked on a text field (red crosshairs + coordinates show the click location). 
Suggest what to enter based on their clipboard.

## Clipboard format
Labeled images [c1], [c2], etc. with text descriptions.

## Screenshot format  
Red dotted crosshairs mark where the user clicked. Axis labels show normalized 0-1000 coordinates.

## Response format
Brief reasoning (1-2 sentences), then JSON:

Reasoning: [what field is at the crosshairs, what info it needs]
Answer: {"suggestion": "text", "source": "c1", "bbox": [x1, y1, x2, y2]}

- suggestion: text to fill, or null if not in clipboard
- source: clipboard image label (e.g. "c1") if from image, else null  
- bbox: [x1,y1,x2,y2] normalized 0-1000 if source is image, else null

If not found: {"suggestion": null, "source": null, "bbox": null}

NEVER guess."""

COMPACT_PROMPT = """Summarize the user's recent actions into a brief paragraph. Extract key information relevant to understanding what the user is doing and what they might need to fill in next. Be concise."""

# Session state: {session_id: {"clipboard": [...], "actions": [], "tokens": 0}}
sessions: dict[str, dict] = {}

# Global LLM instance
llm: LLM | None = None


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
    reasoning: str | None = None
    suggestion: str | None = None
    source: str | None = None
    bbox: list[int] | None = None


def get_session(sid: str) -> dict:
    if sid not in sessions:
        sessions[sid] = {"clipboard": [], "actions": [], "tokens": 0}
    return sessions[sid]


def parse_reasoning_response(raw: str) -> dict:
    """Parse 'Reasoning: ... Answer: {...}' format into structured dict."""
    result = {"reasoning": None, "suggestion": None, "source": None, "bbox": None}

    # Extract reasoning
    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=Answer:|$)", raw, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract JSON from Answer: line
    answer_match = re.search(r"Answer:\s*(\{.+\})", raw, re.DOTALL)
    if answer_match:
        try:
            parsed = json.loads(answer_match.group(1))
            result["suggestion"] = parsed.get("suggestion")
            result["source"] = parsed.get("source")
            result["bbox"] = parsed.get("bbox")
        except json.JSONDecodeError:
            pass
    else:
        # Fallback: try to find any JSON in the response
        json_match = re.search(r"\{[^{}]*\}", raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                result["suggestion"] = parsed.get("suggestion")
                result["source"] = parsed.get("source")
                result["bbox"] = parsed.get("bbox")
            except json.JSONDecodeError:
                pass

    return result


def call_model(
    sid: str | None,
    content: list[dict],
    max_tokens: int = MAX_PREDICT_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, int]:
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    outputs = llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    prompt_tokens = len(outputs[0].prompt_token_ids)

    # Track tokens in session
    if sid and sid in sessions:
        sessions[sid]["tokens"] = prompt_tokens

    return generated_text, prompt_tokens


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype=DTYPE,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 100},
        mm_processor_kwargs={"min_pixels": 784, "max_pixels": 2073600},
    )
    # Warmup
    call_model(None, [{"type": "text", "text": "hi"}], max_tokens=1)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": llm is not None, "model": MODEL}


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
    if not session["actions"]:
        session["actions"].append({"type": "text", "text": "Recent Actions:"})
    session["actions"].extend(req.blocks)

    # Sliding window: keep header + last MAX_ACTIONS pairs (each pair = text label + image)
    if MAX_ACTIONS > 0:
        header = session["actions"][:1]  # "Recent Actions:" text block
        pairs = session["actions"][1:]  # action pairs after header
        max_blocks = MAX_ACTIONS * 2
        if len(pairs) > max_blocks:
            pairs = pairs[-max_blocks:]
        session["actions"] = header + pairs

    content = session["clipboard"] + session["actions"]
    _, prompt_tokens = call_model(req.session_id, content, max_tokens=1)
    return ActionResponse(session_id=req.session_id, prompt_tokens=prompt_tokens)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: SessionRequest):
    session = get_session(req.session_id)
    content = session["clipboard"] + session["actions"]
    raw, prompt_tokens = call_model(req.session_id, content, max_tokens=300)
    parsed = parse_reasoning_response(raw)
    return PredictResponse(
        session_id=req.session_id,
        prompt_tokens=prompt_tokens,
        raw=raw,
        reasoning=parsed["reasoning"],
        suggestion=parsed["suggestion"],
        source=parsed["source"],
        bbox=parsed["bbox"],
    )


@app.post("/compact", response_model=CompactResponse)
async def compact(req: SessionRequest):
    session = get_session(req.session_id)
    content = (
        session["clipboard"]
        + session["actions"]
        + [{"type": "text", "text": f"\n\n{COMPACT_PROMPT}"}]
    )
    summary, _ = call_model(
        req.session_id,
        content,
        max_tokens=MAX_COMPACT_TOKENS,
        system_prompt=COMPACT_PROMPT,
    )
    session["actions"] = [
        {"type": "text", "text": f"...\n[Previous Actions Summary]\n{summary}"}
    ]
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
