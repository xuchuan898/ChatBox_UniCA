"""Document selection and index management routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.dependencies import get_app_state
from api.services.doc_manager import (
    list_available_docs,
    build_and_persist_multi_doc_index,
    load_multi_doc_index,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


class SelectDocsRequest(BaseModel):
    selected: list[str]


@router.get("/list")
async def list_docs() -> list[dict]:
    """Return all available documents in docs/chroma."""
    return list_available_docs()


@router.get("/active")
async def get_active_docs() -> dict:
    """Return currently selected document paths, if any."""
    state = get_app_state()
    meta = getattr(state, "_current_doc_meta", None)
    if meta:
        return {"selected": meta.get("doc_files", [])}
    return {"selected": []}


@router.post("/select")
async def select_and_build(body: SelectDocsRequest) -> dict[str, Any]:
    """Select documents and build/load their index. Hot-reloads the retriever."""
    if not body.selected:
        return {"message": "No documents selected", "doc_count": 0, "chunk_count": 0}

    state = get_app_state()

    # Try loading existing index first
    idx = load_multi_doc_index(body.selected)
    if idx is None:
        # Build from scratch
        logger.info("Building new multi-doc index from %d files", len(body.selected))
        idx = build_and_persist_multi_doc_index(body.selected)
    else:
        logger.info("Loaded existing multi-doc index from disk")

    # Hot-reload the retriever in AppState
    retrieval_cfg = state.config.get("retrieval", {})
    qe_cfg = state.config.get("query_expansion", {})

    from core.query_expander import QueryExpander, OllamaRewriter, PassthroughRewriter

    rewriter_type = qe_cfg.get("rewriter", "ollama")
    if rewriter_type == "ollama":
        rewriter = OllamaRewriter(model_name=qe_cfg.get("model_name", "qwen2.5:32b"))
    else:
        rewriter = PassthroughRewriter()

    query_expander = QueryExpander(
        rewriter=rewriter,
        num_paraphrases=qe_cfg.get("num_paraphrases", 1),
        add_translation=qe_cfg.get("add_translation", True),
        source_lang=qe_cfg.get("source_lang", "auto"),
        target_lang_for_translation=qe_cfg.get("target_lang_for_translation", "auto"),
    )

    from core.retriever import HybridRetriever
    from core.reranker import CrossEncoderReranker

    state.retriever = HybridRetriever(
        vectordb=idx["vectordb"],
        bm25_index=idx["bm25_index"],
        tokenizer=idx["tokenizer"],
        corpus=idx["corpus"],
        reranker=state.reranker,
        weight_vec=retrieval_cfg.get("weight_vec", 1.25),
        weight_bm25=retrieval_cfg.get("weight_bm25", 0.75),
        original_weight=retrieval_cfg.get("original_weight", 1.0),
        paraphrase_weight=retrieval_cfg.get("paraphrase_weight", 1.2),
        translation_weight=retrieval_cfg.get("translation_weight", 0.85),
        rerank_candidates=retrieval_cfg.get("rerank_candidates", 30),
        dynamic_topk_ratio=retrieval_cfg.get("dynamic_topk_ratio", 0.5),
        query_expander=query_expander,
        query_expansion_enabled=qe_cfg.get("enabled", False),
        multi_turn_enabled=qe_cfg.get("multi_turn", {}).get("enabled", True),
        max_history_turns=qe_cfg.get("multi_turn", {}).get("max_history_turns", 5),
    )
    state._current_doc_meta = idx["meta"]

    meta = idx["meta"]
    logger.info(
        "Hot-reloaded retriever: %d docs, %d chunks",
        meta.get("doc_count", 0),
        meta.get("chunk_count", 0),
    )

    return {
        "message": "Index built and retriever hot-reloaded",
        "doc_count": meta.get("doc_count", 0),
        "chunk_count": meta.get("chunk_count", 0),
        "doc_files": meta.get("doc_files", []),
    }