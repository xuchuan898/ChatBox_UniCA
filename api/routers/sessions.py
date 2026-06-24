"""Session management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_session_service
from services.session_service import SessionService

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("/new")
async def create_session(
    session_service: SessionService = Depends(get_session_service),
) -> dict:
    """Create a new session."""
    session, session_id = session_service.get_or_create()
    return {"session_id": session_id}


@router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> dict:
    """Clear a specific session's memory."""
    session_service.clear(session_id)
    return {"message": f"Session {session_id} cleared"}