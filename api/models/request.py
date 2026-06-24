from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID, auto-generated if empty")
    overrides: Optional[dict[str, Any]] = Field(
        None,
        description="Runtime parameter overrides, e.g. top_k, weight_vec, weight_bm25, rerank_alpha, rerank_candidates",
    )