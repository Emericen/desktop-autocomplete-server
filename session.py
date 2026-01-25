import os

import openai
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
API_KEY = os.getenv("API_KEY", "EMPTY")
MAX_PREDICT_TOKENS = int(os.getenv("MAX_PREDICT_TOKENS", "20"))
MAX_COMPACT_TOKENS = int(os.getenv("MAX_COMPACT_TOKENS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "100000"))


class Action(BaseModel):
    id: str
    timestamp: int

    def to_llm_content_blocks(self) -> list[dict]:
        raise NotImplementedError("Subclasses must implement this method")


class TypingAction(Action):
    text: str
    screenshot: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [
            {"type": "text", "text": f'Typed: "{self.text}"'},
            {"type": "image_url", "image_url": {"url": self.screenshot}},
        ]


class MouseClickAction(Action):
    button: str
    x: float
    y: float
    screenshot: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [
            {"type": "text", "text": f"{self.button.capitalize()} click"},
            {"type": "image_url", "image_url": {"url": self.screenshot}},
        ]


class MouseDragAction(Action):
    button: str
    start_x: float = Field(alias="startX")
    start_y: float = Field(alias="startY")
    end_x: float = Field(alias="endX")
    end_y: float = Field(alias="endY")
    screenshot: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [
            {"type": "text", "text": f"Dragged with {self.button} mouse button"},
            {"type": "image_url", "image_url": {"url": self.screenshot}},
        ]


class ScrollAction(Action):
    x: float
    y: float
    screenshot: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [
            {"type": "text", "text": "Scrolled"},
            {"type": "image_url", "image_url": {"url": self.screenshot}},
        ]


class HotkeyAction(Action):
    modifiers: list[str]
    key: str

    def to_llm_content_blocks(self) -> list[dict]:
        combo = " + ".join(m.capitalize() for m in self.modifiers) + f" + {self.key}"
        return [{"type": "text", "text": f"Hit `{combo}`"}]


class SpecialKeyAction(Action):
    key: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [{"type": "text", "text": f"Hit `{self.key}`"}]


class AutocompleteAction(Action):
    text: str

    def to_llm_content_blocks(self) -> list[dict]:
        return [{"type": "text", "text": f'Accepted: "{self.text}"'}]


def parse_action(data: dict) -> Action:
    action_type = data.get("type")
    if action_type == "typing":
        return TypingAction(**data)
    elif action_type == "mouse_click":
        return MouseClickAction(**data)
    elif action_type == "mouse_drag":
        return MouseDragAction(**data)
    elif action_type == "scroll":
        return ScrollAction(**data)
    elif action_type == "hotkey":
        return HotkeyAction(**data)
    elif action_type == "special_key":
        return SpecialKeyAction(**data)
    elif action_type == "autocomplete":
        return AutocompleteAction(**data)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


SYSTEM_PROMPT = """You are a helpful assistant that helps users with their tasks."""
COMPACT_PROMPT = """Summarize everything you see so far into 2 or 3 sentences."""


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = openai.OpenAI(api_key=API_KEY, base_url=VLLM_BASE_URL)
        self.actions: list[Action] = []
        self.content_blocks: list[dict] = []
        self.context_length = 0

    def predict(self, action: dict) -> tuple[str, bool]:
        action = parse_action(action)
        self.content_blocks.extend(action.to_llm_content_blocks())
        self.actions.append(action)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.content_blocks},
            ],
            max_tokens=MAX_PREDICT_TOKENS,
            temperature=TEMPERATURE,
        )
        self.context_length += response.usage.prompt_tokens
        prediction = response.choices[0].message.content
        needs_compact = self.context_length > MAX_MODEL_LEN
        if needs_compact:
            self.compact()
        return prediction, needs_compact

    def compact(self):
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.content_blocks},
                {"role": "user", "content": COMPACT_PROMPT},
            ],
            max_tokens=MAX_COMPACT_TOKENS,
            temperature=TEMPERATURE,
        )
        summary = response.choices[0].message.content
        self.content_blocks = [{"type": "text", "text": summary}]
