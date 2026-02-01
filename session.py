import os
import json
import openai
from typing import Any

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
API_KEY = os.getenv("API_KEY", "EMPTY")
MAX_PREDICT_TOKENS = int(os.getenv("MAX_PREDICT_TOKENS", "100"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_ACTION_BLOCKS = int(
    os.getenv("MAX_ACTION_BLOCKS", "15")
)  # ~5 actions × 2-3 blocks each

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


class Session:
    """Stateful session for streaming context + actions to LLM.

    Everything is OpenAI content blocks — no conversion needed.
    """

    def __init__(
        self,
        session_id: str,
        base_url: str = VLLM_BASE_URL,
        model: str = MODEL,
        api_key: str = API_KEY,
    ):
        self.session_id = session_id
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.context: list[dict] = []  # OpenAI content blocks (static)
        self.actions: list[dict] = []  # OpenAI content blocks (dynamic)

    def set_context(self, blocks: list[dict]) -> None:
        self.context = blocks

    def add_action(self, blocks: list[dict]) -> None:
        self.actions.extend(blocks)
        while len(self.actions) > MAX_ACTION_BLOCKS:
            self.actions.pop(0)
        self._request_model(max_tokens=1)  # warm cache

    def predict(self) -> dict[str, Any]:
        response = self._request_model()
        try:
            response_json = json.loads(response.choices[0].message.content)
            return {
                "suggestion": response_json.get("suggestion", ""),
                "source": response_json.get("source", None),
                "bbox": response_json.get("bbox", None),
            }
        except json.JSONDecodeError:
            print(f"Error: {response.choices[0].message.content}")
            return {"suggestion": "", "source": None, "bbox": None}

    def _request_model(
        self, max_tokens: int = MAX_PREDICT_TOKENS
    ) -> openai.ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.context + self.actions},
            ],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
        )

    def clear(self) -> None:
        self.context = []
        self.actions = []
