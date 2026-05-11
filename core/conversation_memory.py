"""Short-term and long-term conversation memory."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from langchain_ollama import ChatOllama


class ConversationMemory:
    """Conversation memory manager."""

    def __init__(self, rounds: int = 5, memory_file: str | Path = "./index_store/user_memory.json", model_name: str = "gemma3:1b"):
        self.rounds = max(1, rounds)
        self.short_term = deque(maxlen=self.rounds)
        self.path = Path(memory_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term = self._load_long_term()
        self.llm = ChatOllama(model=model_name, temperature=0.0, seed=42, validate_model_on_init=True, num_predict=128)

    def _load_long_term(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_long_term(self) -> None:
        self.path.write_text(json.dumps(self.long_term, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_turn(self, user: str, assistant: str) -> None:
        self.short_term.append({"user": user, "assistant": assistant})
        summary = self._extract_memory(user, assistant)
        if summary:
            self.long_term.append(summary)
            self.long_term = self.long_term[-200:]
            self._save_long_term()

    def build_short_prompt(self) -> str:
        if not self.short_term:
            return ""
        lines = ["Recent conversation:"]
        for turn in self.short_term:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def build_long_prompt(self, max_items: int = 5) -> str:
        if not self.long_term:
            return ""
        lines = ["User memory facts:"]
        for item in self.long_term[-max_items:]:
            lines.append(f"- {item.get('fact', '')}")
        return "\n".join(lines)

    def _extract_memory(self, user: str, assistant: str) -> dict | None:
        prompt = (
            "Extract one stable user-related memory fact from the dialogue.\n"
            "Return JSON object {'fact': '<text>'} or {} if nothing stable.\n"
            f"User: {user}\nAssistant: {assistant}"
        )
        try:
            raw = str(self.llm.invoke(prompt).content).strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return None
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict) and obj.get("fact"):
                return {"fact": str(obj["fact"])}
        except Exception:
            return None
        return None
