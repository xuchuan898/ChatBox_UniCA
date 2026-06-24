"""Index status and rebuild routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends

from api.dependencies import get_app_state
from api.models.response import IndexStatusResponse
from core.indexer import load_index_meta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/index", tags=["Index"])


@router.get("/status", response_model=IndexStatusResponse)
async def index_status() -> IndexStatusResponse:
    """Get the current index status."""
    state = get_app_state()
    cfg = state.config
    persist_dir = Path(cfg.get("indexing", {}).get("persist_dir", "./index_store"))
    meta = load_index_meta(persist_dir)

    if meta is None:
        return IndexStatusResponse(
            is_ready=False,
            doc_count=0,
            vector_count=0,
            last_build_time=None,
            embedding_model="",
        )

    return IndexStatusResponse(
        is_ready=True,
        doc_count=meta.get("doc_count", 0),
        vector_count=meta.get("chunk_count", 0),
        last_build_time=meta.get("built_at"),
        embedding_model=meta.get("embedding_model", ""),
    )


@router.post("/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks) -> dict:
    """Rebuild the index asynchronously."""
    state = get_app_state()
    cfg = state.config
    persist_dir = cfg.get("indexing", {}).get("persist_dir", "./index_store")
    doc_file = cfg.get("ragas_experiment", {}).get("doc_file", "./docs/chroma/master.md")

    def _rebuild():
        from build_index import main as build_main
        import sys
        import argparse

        args = [
            "build_index.py",
            "--doc-file", doc_file,
            "--persist-dir", persist_dir,
            "--force-rebuild",
        ]
        sys.argv = args
        build_main()

    background_tasks.add_task(_rebuild)
    return {"message": "Index rebuild task submitted"}


@router.post("/update")
async def update_index(background_tasks: BackgroundTasks) -> dict:
    """Incrementally update the index."""
    state = get_app_state()
    cfg = state.config
    persist_dir = cfg.get("indexing", {}).get("persist_dir", "./index_store")
    doc_file = cfg.get("ragas_experiment", {}).get("doc_file", "./docs/chroma/master.md")

    def _update():
        from core.indexer import ensure_index
        _ = ensure_index(
            doc_file=doc_file,
            url_file=None,
            persist_dir=persist_dir,
            force_rebuild=False,
        )

    background_tasks.add_task(_update)
    return {"message": "Index update task submitted"}