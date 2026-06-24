"""FastAPI route registrations."""

from api.routers import chat, config, index, sessions

__all__ = ["chat", "config", "index", "sessions"]