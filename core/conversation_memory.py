"""Short-term conversation memory with a disabled long-term memory interface."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path


class ConversationMemory:
    """Conversation memory manager."""

    def __init__(
        self,
        rounds: int = 5,
        memory_file: str | Path = "./index_store/user_memory.json",
        model_name: str = "gemma3:4b",
        long_term_enabled: bool = False,
    ):
        _ = model_name, long_term_enabled
        self.rounds = max(1, rounds)
        self.short_term = deque(maxlen=self.rounds)
        self.path = Path(memory_file)
        # Long-term memory is intentionally disabled. The constructor argument
        # remains as a config/API placeholder for a future re-enable.
        self.long_term_enabled = False
        self.long_term = []

    def _load_long_term(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_long_term(self) -> None:
        if not self.long_term_enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.long_term, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_turn(self, user: str, assistant: str) -> None:
        self.short_term.append({"user": user, "assistant": assistant})

    def build_short_prompt(self) -> str:
        if not self.short_term:
            return ""
        lines = ["Recent conversation:"]
        for turn in self.short_term:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def build_long_prompt(self, max_items: int = 5) -> str:
        _ = max_items
        return ""

    def _extract_memory(self, user: str, assistant: str) -> dict | None:
        _ = user, assistant
        return None

    def _build_long_prompt_legacy(self, max_items: int = 5) -> str:
        if not self.long_term:
            return ""
        lines = ["User memory facts:"]
        for item in self.long_term[-max_items:]:
            lines.append(f"- {item.get('fact', '')}")
        return "\n".join(lines)
