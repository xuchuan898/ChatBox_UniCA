"""Chat Q&A routes (including SSE streaming)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from api.dependencies import get_rag_service
from api.models.request import ChatRequest
from api.models.response import ChatResponse
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Non-streaming chat endpoint (JSON response)."""
    return await rag.chat(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """SSE streaming chat endpoint."""
    generator = rag.chat_stream(request)

    async def event_stream():
        async for event in generator:
            yield f"{event}\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )