"""Debug retrieval endpoint — returns full retrieval trace without generation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from langchain_core.documents import Document

from api.dependencies import get_app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/debug", tags=["Debug"])


def _short_source(doc: Document) -> str:
    src = str(doc.metadata.get("source", "unknown"))
    return src.split("/")[-1].split("\\")[-1]


def _compact_preview(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


@router.get("/retrieve")
async def debug_retrieve(
    query: str = Query(..., min_length=1, description="Query to test retrieval"),
) -> dict[str, Any]:
    """Run retrieval pipeline and return full trace without generating an answer."""
    state = get_app_state()
    if not state.retriever:
        return {"error": "Retriever not initialized"}

    docs = state.retriever.retrieve(query)

    # Rerank with scores for display
    if docs and state.retriever.reranker:
        reranked = state.retriever.reranker.rerank(
            query,
            docs,
            top_n=len(docs),
            return_scores=True,
        )
        for doc, score, _, _ in reranked:
            doc.metadata["_rerank_score"] = score
        docs = [doc for doc, _, _, _ in reranked]

    reranked = []
    for i, doc in enumerate(docs):
        reranked.append({
            "rank": i + 1,
            "score": round(doc.metadata.get("_rerank_score", 0.0), 4),
            "source": _short_source(doc),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "content_preview": _compact_preview(doc.page_content),
        })

    return {
        "query": query,
        "final_doc_count": len(docs),
        "reranked": reranked,
    }