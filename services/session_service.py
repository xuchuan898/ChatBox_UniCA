from __future__ import annotations

from collections import deque
from typing import Optional


class ConversationSession:
    """Per-session conversation memory using a deque of recent turns."""

    def __init__(self, session_id: str, max_rounds: int = 5):
        self.session_id = session_id
        self.max_rounds = max_rounds
        self._history: deque[dict[str, str]] = deque(maxlen=max_rounds)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"user": user_msg, "assistant": assistant_msg})

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()


class SessionService:
    """In-memory session management pool."""

    def __init__(self, max_rounds: int = 5):
        self._max_rounds = max_rounds
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> tuple[ConversationSession, str]:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], session_id
        import uuid

        sid = session_id or uuid.uuid4().hex[:12]
        session = ConversationSession(sid, max_rounds=self._max_rounds)
        self._sessions[sid] = session
        return session, sid

    def list_sessions(self) -> list[dict]:
        now = __import__("datetime").datetime.now()
        results = []
        for sid, sess in self._sessions.items():
            history = sess.get_history()
            last_activity = history[-1]["assistant"][:80] if history else ""
            results.append({
                "session_id": sid,
                "turn_count": len(history),
                "last_question": history[-1]["user"][:120] if history else "",
                "last_activity_preview": last_activity,
            })
        return sorted(results, key=lambda x: x["session_id"])

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def clear_all(self) -> None:
        self._sessions.clear()