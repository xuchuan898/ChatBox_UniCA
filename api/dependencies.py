"""Global singleton dependency injection."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config_loader import load_config
from core.generator import AnswerGenerator
from core.indexer import ensure_index, load_index_meta, load_persisted_index
from core.query_expander import (
    OllamaRewriter,
    PassthroughRewriter,
    QueryExpander,
)
from core.reranker import CrossEncoderReranker
from core.retriever import HybridRetriever
from core.semantic_cache import SemanticCache
from services.session_service import SessionService

logger = logging.getLogger(__name__)


class AppState:
    """Application global state holding all singleton components."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.retriever: HybridRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.generator: AnswerGenerator | None = None
        self.cache: SemanticCache | None = None
        self.session_service: SessionService | None = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        cfg = self.config
        persist_dir = Path(cfg.get("indexing", {}).get("persist_dir", "./index_store"))

        # ---------- Index Loading ----------
        doc_file = cfg.get("ragas_experiment", {}).get("doc_file", "./docs/chroma/master.md")

        # Try loading meta from the base persist_dir first, then search subdirectories
        meta = load_index_meta(persist_dir)
        actual_persist_dir = persist_dir

        if meta is None:
            subdirs = sorted(persist_dir.glob("*/index_meta.json"))
            if subdirs:
                actual_persist_dir = subdirs[0].parent
                meta = load_index_meta(actual_persist_dir)
                logger.info("Found index in subdirectory: %s", actual_persist_dir)

        if meta:
            logger.info("Loading persisted index from %s", actual_persist_dir)
            state = load_persisted_index(actual_persist_dir)
        else:
            logger.info("No index found, building from %s", doc_file)
            state = ensure_index(
                doc_file=doc_file,
                url_file=None,
                persist_dir=persist_dir,
                force_rebuild=False,
            )

        vectordb = state["vectordb"]
        bm25_index = state["bm25_index"]
        tokenizer = state["tokenizer"]
        corpus = state["corpus"]
        idx_meta = state.get("meta", {})

        logger.info("Index loaded: %s docs, %s chunks", idx_meta.get("doc_count"), idx_meta.get("chunk_count"))

        # ---------- Reranker ----------
        retrieval_cfg = cfg.get("retrieval", {})
        self.reranker = CrossEncoderReranker(
            alpha=retrieval_cfg.get("rerank_alpha", 0.5),
            multi_variant_enabled=retrieval_cfg.get("multi_variant_rerank_enabled", False),
        )

        # ---------- Query Expander ----------
        qe_cfg = cfg.get("query_expansion", {})
        rewriter_type = qe_cfg.get("rewriter", "ollama")
        if rewriter_type == "ollama":
            rewriter = OllamaRewriter(
                model_name=qe_cfg.get("model_name", "qwen2.5:32b"),
            )
        else:
            rewriter = PassthroughRewriter()

        query_expander = QueryExpander(
            rewriter=rewriter,
            num_paraphrases=qe_cfg.get("num_paraphrases", 1),
            add_translation=qe_cfg.get("add_translation", True),
            source_lang=qe_cfg.get("source_lang", "auto"),
            target_lang_for_translation=qe_cfg.get("target_lang_for_translation", "auto"),
        )

        # ---------- Retriever ----------
        self.retriever = HybridRetriever(
            vectordb=vectordb,
            bm25_index=bm25_index,
            tokenizer=tokenizer,
            corpus=corpus,
            reranker=self.reranker,
            weight_vec=retrieval_cfg.get("weight_vec", 1.25),
            weight_bm25=retrieval_cfg.get("weight_bm25", 0.75),
            original_weight=retrieval_cfg.get("original_weight", 1.0),
            paraphrase_weight=retrieval_cfg.get("paraphrase_weight", 1.2),
            translation_weight=retrieval_cfg.get("translation_weight", 0.85),
            rerank_candidates=retrieval_cfg.get("rerank_candidates", 30),
            dynamic_topk_ratio=retrieval_cfg.get("dynamic_topk_ratio", 0.1),
            disable_rerank=retrieval_cfg.get("disable_rerank", False),
            query_expander=query_expander,
            query_expansion_enabled=qe_cfg.get("enabled", False),
            multi_turn_enabled=qe_cfg.get("multi_turn", {}).get("enabled", True),
            max_history_turns=qe_cfg.get("multi_turn", {}).get("max_history_turns", 5),
        )

        # ---------- Generator ----------
        gen_cfg = cfg.get("generation", {})
        self.generator = AnswerGenerator(
            model_name=gen_cfg.get("model_name", "qwen2.5:32b"),
            temperature=gen_cfg.get("temperature", 0.0),
            num_predict=gen_cfg.get("num_predict", 256),
        )

        # ---------- Semantic Cache ----------
        cache_cfg = cfg.get("cache", {})
        if cache_cfg.get("enabled", False):
            self.cache = SemanticCache(
                path=Path(persist_dir) / "semantic_cache.json",
                threshold=cache_cfg.get("threshold", 0.92),
                ttl=cache_cfg.get("ttl", 86400),
            )
        else:
            self.cache = None

        # ---------- Session Service ----------
        memory_cfg = cfg.get("memory", {})
        self.session_service = SessionService(
            max_rounds=memory_cfg.get("short_term_rounds", 5),
        )

        self._initialized = True
        logger.info("AppState initialized successfully")


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    """Get the global singleton AppState."""
    config = load_config()
    state = AppState(config)
    state.initialize()
    return state


# ---- Convenience Depends functions ----
def get_retriever() -> HybridRetriever:
    return get_app_state().retriever


def get_reranker() -> CrossEncoderReranker:
    return get_app_state().reranker


def get_generator() -> AnswerGenerator:
    return get_app_state().generator


def get_cache() -> SemanticCache | None:
    return get_app_state().cache


def get_session_service() -> SessionService:
    return get_app_state().session_service


def get_rag_service():
    """Lazy import to avoid circular dependency."""
    from services.rag_service import RAGService

    state = get_app_state()
    return RAGService(
        retriever=state.retriever,
        reranker=state.reranker,
        generator=state.generator,
        cache=state.cache,
        session_service=state.session_service,
        config=state.config,
    )