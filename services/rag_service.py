"""Full RAG pipeline orchestration service wrapping retrieval, reranking, and generation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

from langchain_core.documents import Document

from api.models.request import ChatRequest
from api.models.response import ChatResponse, SourceDoc
from core.config_loader import apply_cli_overrides
from core.generator import AnswerGenerator
from core.reranker import CrossEncoderReranker
from core.retriever import HybridRetriever
from core.semantic_cache import SemanticCache
from services.session_service import SessionService

logger = logging.getLogger(__name__)


def _doc_to_source(doc: Document) -> SourceDoc:
    return SourceDoc(
        content_preview=" ".join(doc.page_content.split())[:220],
        source=str(doc.metadata.get("source", "unknown")).split("/")[-1].split("\\")[-1],
        chunk_id=str(doc.metadata.get("chunk_id", "")),
        score=float(doc.metadata.get("_rerank_score", 0.0)),
    )


def _stage_event(stage: str, data: dict) -> str:
    payload = {"stage": stage, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class RAGService:
    """Full RAG pipeline orchestration service."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: AnswerGenerator,
        cache: SemanticCache | None,
        session_service: SessionService,
        config: dict[str, Any],
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.cache = cache
        self.session_service = session_service
        self.config = config

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return await asyncio.to_thread(self._sync_execute_pipeline, request)

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """SSE streaming response that pushes stage events progressively."""
        t0 = time.perf_counter()

        session, session_id = self.session_service.get_or_create(request.session_id)
        history = session.get_history()
        overrides = request.overrides or {}
        cfg = apply_cli_overrides(self.config, overrides) if overrides else self.config

        yield _stage_event("session", {"session_id": session_id})

        # 1) Semantic cache check
        cache_cfg = cfg.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", False) and self.cache is not None
        if cache_enabled:
            yield _stage_event("cache_check", {"status": "checking"})
            cached = await asyncio.to_thread(self.cache.get, request.query)
            if cached:
                elapsed = time.perf_counter() - t0
                resp = ChatResponse(
                    session_id=session_id,
                    answer=str(cached.get("answer", "")),
                    sources=[SourceDoc(**d) for d in cached.get("context_docs", [])],
                    elapsed_seconds=round(elapsed, 3),
                    from_cache=True,
                )
                yield _stage_event("done", resp.model_dump(mode="json"))
                return

        # 2) Retrieval
        yield _stage_event("retrieving", {"status": "started"})
        retrieval_cfg = cfg.get("retrieval", {})
        self.retriever.weight_vec = retrieval_cfg.get("weight_vec", 1.25)
        self.retriever.weight_bm25 = retrieval_cfg.get("weight_bm25", 0.75)
        self.retriever.rerank_candidates = retrieval_cfg.get("rerank_candidates", 30)
        self.reranker.alpha = retrieval_cfg.get("rerank_alpha", 0.5)

        docs = await asyncio.to_thread(self.retriever.retrieve, request.query, history)
        yield _stage_event("retrieved", {"doc_count": len(docs)})

        # 3) Generation
        yield _stage_event("generating", {"status": "started"})
        answer = await asyncio.to_thread(self.generator.answer, request.query, docs, "")
        elapsed = time.perf_counter() - t0

        # 4) Write-back cache
        if cache_enabled:
            source_docs = [_doc_to_source(d) for d in docs]
            await asyncio.to_thread(
                self.cache.put,
                request.query,
                answer,
                [s.model_dump() for s in source_docs],
            )

        # 5) Write-back session memory
        session.add_turn(request.query, answer)

        resp = ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=[_doc_to_source(d) for d in docs],
            elapsed_seconds=round(elapsed, 3),
            from_cache=False,
        )
        yield _stage_event("done", resp.model_dump(mode="json"))

    def _sync_execute_pipeline(self, request: ChatRequest) -> ChatResponse:
        """Synchronous core pipeline (executed inside asyncio.to_thread)."""
        t0 = time.perf_counter()

        session, session_id = self.session_service.get_or_create(request.session_id)
        history = session.get_history()
        overrides = request.overrides or {}
        cfg = apply_cli_overrides(self.config, overrides) if overrides else self.config

        # 1) Semantic cache check
        cache_cfg = cfg.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", False) and self.cache is not None
        if cache_enabled:
            cached = self.cache.get(request.query)
            if cached:
                elapsed = time.perf_counter() - t0
                return ChatResponse(
                    session_id=session_id,
                    answer=str(cached.get("answer", "")),
                    sources=[SourceDoc(**d) for d in cached.get("context_docs", [])],
                    elapsed_seconds=round(elapsed, 3),
                    from_cache=True,
                )

        # 2) Query expansion & retrieval param overrides
        retrieval_cfg = cfg.get("retrieval", {})
        self.retriever.weight_vec = retrieval_cfg.get("weight_vec", 1.25)
        self.retriever.weight_bm25 = retrieval_cfg.get("weight_bm25", 0.75)
        self.retriever.rerank_candidates = retrieval_cfg.get("rerank_candidates", 30)
        self.reranker.alpha = retrieval_cfg.get("rerank_alpha", 0.5)

        # 3) Hybrid retrieval
        docs = self.retriever.retrieve(request.query, history=history)

        # 4) Rerank and inject scores
        if docs:
            queries = [request.query]
            if self.retriever.query_expansion_enabled and self.retriever.query_expander:
                qe = self.retriever.query_expander
                queries, _ = qe.expand(request.query, history=history)
            reranked = self.reranker.rerank(
                request.query,
                docs,
                top_n=len(docs),
                return_scores=True,
                query_variants=queries,
            )
            for doc, score, _, _ in reranked:
                doc.metadata["_rerank_score"] = score
            docs = [doc for doc, _, _, _ in reranked]

        # 5) Generate answer
        answer = self.generator.answer(request.query, docs, "")
        elapsed = time.perf_counter() - t0

        # 6) Write-back cache
        if cache_enabled:
            source_docs = [_doc_to_source(d) for d in docs]
            self.cache.put(request.query, answer, [s.model_dump() for s in source_docs])

        # 7) Write-back session memory
        session.add_turn(request.query, answer)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=[_doc_to_source(d) for d in docs],
            elapsed_seconds=round(elapsed, 3),
            from_cache=False,
        )