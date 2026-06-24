from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    content_preview: str = Field(..., description="Document content preview")
    source: str = Field(..., description="Source filename")
    chunk_id: str = Field("", description="Chunk ID")
    score: float = Field(0.0, description="Relevance score")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDoc] = Field(default_factory=list, description="Referenced source documents")
    elapsed_seconds: float = Field(0.0, description="Elapsed time in seconds")
    from_cache: bool = Field(False, description="Whether the response came from cache")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class IndexStatusResponse(BaseModel):
    is_ready: bool = Field(..., description="Whether the index is ready")
    doc_count: int = Field(0, description="Number of documents")
    vector_count: int = Field(0, description="Number of vectors/chunks")
    last_build_time: Optional[str] = Field(None, description="Last build timestamp")
    embedding_model: str = Field("", description="Embedding model name")