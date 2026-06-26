"""Hybrid retrieval (vector + BM25 + RRF) with query expansion support."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .query_expander import QueryExpander, detect_language
from .reranker import CrossEncoderReranker


def _short_source(doc: Document) -> str:
    src = str(doc.metadata.get("source", "unknown"))
    return src.split("/")[-1].split("\\")[-1]


def _compact_preview(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("`", "'")


class HybridRetriever:
    """Hybrid retrieval pipeline with optional query expansion."""

    def __init__(
        self,
        vectordb: Any,
        bm25_index: Any,
        tokenizer: Any,
        corpus: list[Document],
        reranker: CrossEncoderReranker,
        weight_vec: float = 1.25,
        weight_bm25: float = 0.75,
        original_weight: float = 1.0,
        paraphrase_weight: float = 1.2,
        translation_weight: float = 0.85,
        rerank_candidates: int = 30,
        query_expander: QueryExpander | None = None,
        query_expansion_enabled: bool = False,
        multi_turn_enabled: bool = True,
        max_history_turns: int = 5,
        dynamic_topk_ratio: float = 0.1,
        debug: bool = False,
    ):
        self.vectordb = vectordb
        self.base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": rerank_candidates, "fetch_k": max(rerank_candidates + 20, rerank_candidates), "lambda_mult": 0.7})
        self.bm25_index = bm25_index
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.reranker = reranker
        self.weight_vec = weight_vec
        self.weight_bm25 = weight_bm25
        self.original_weight = original_weight
        self.paraphrase_weight = paraphrase_weight
        self.translation_weight = translation_weight
        self.rerank_candidates = rerank_candidates
        self.query_expander = query_expander
        self.query_expansion_enabled = query_expansion_enabled
        self.multi_turn_enabled = multi_turn_enabled
        self.max_history_turns = max(0, int(max_history_turns))
        self.dynamic_topk_ratio = dynamic_topk_ratio
        self.debug = debug
        self.retrieval_cache: dict[str, list[Document]] = {}

    def _bm25_docs(self, query_text: str) -> list[Document]:
        q_tokens = self.tokenizer.tokenize(query_text)
        bm25_results, _ = self.bm25_index.retrieve(q_tokens, k=min(self.rerank_candidates, len(self.corpus)))
        indices = bm25_results[0] if getattr(bm25_results, "ndim", 1) == 2 else bm25_results
        return [self.corpus[i] for i in (indices.tolist() if hasattr(indices, "tolist") else list(indices)) if 0 <= i < len(self.corpus)]

    @staticmethod
    def _weighted_rrf(ranked_lists: list[tuple[list[Document], float]], k: int = 60) -> list[tuple[Document, float]]:
        scores = defaultdict(float)
        by_id: dict[int, Document] = {}
        for docs, weight in ranked_lists:
            for rank, doc in enumerate(docs, 1):
                doc_id = str(doc.metadata.get("chunk_id") or hash(doc.page_content))
                scores[doc_id] += weight / (k + rank)
                by_id.setdefault(doc_id, doc)
        return sorted(((by_id[key], val) for key, val in scores.items()), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _dynamic_top_k(reranked_scored: list[tuple[Document, float]], max_k: int = 8, ratio: float = 0.1) -> int:
        if not reranked_scored:
            return 0
        max_score = reranked_scored[0][1]
        threshold = max_score * ratio
        count = sum(1 for _, s in reranked_scored if s >= threshold)
        return min(count, max_k, len(reranked_scored))

    def _normalize_history(
        self,
        history: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        if not self.multi_turn_enabled:
            return []
        source = history if history is not None else chat_history
        if not source:
            return []
        messages: list[dict[str, str]] = []
        for item in source:
            if "role" in item and "content" in item:
                role = str(item.get("role", "")).lower()
                content = " ".join(str(item.get("content", "")).split())
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
                continue
            user = " ".join(str(item.get("user", "")).split())
            assistant = " ".join(str(item.get("assistant", "")).split())
            if user:
                messages.append({"role": "user", "content": user})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        if self.max_history_turns <= 0:
            return []
        return messages[-self.max_history_turns * 2 :]

    def retrieve(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        rewritten: bool = False,
        top_n: int = 8,
    ) -> list[Document]:
        t0 = time.perf_counter()
        normalized_history = self._normalize_history(history=history, chat_history=chat_history)
        cache_key = question if not normalized_history else f"{question}\n__history__={normalized_history!r}"
        if cache_key in self.retrieval_cache:
            if self.debug:
                print(f"[DEBUG][RETRIEVE][CACHE_HIT] q={question!r}")
            return self.retrieval_cache.pop(cache_key)
        queries = [question]

        if self.query_expansion_enabled and self.query_expander:
            t_expand = time.perf_counter()
            queries, rewritten = self.query_expander.expand(question, history=normalized_history)

            if self.debug:
                print(
                    f"[DEBUG][EXPAND] enabled=true variants={len(queries)} "
                    f"rewritten={rewritten} "
                    f"history_messages={len(normalized_history)} seconds={time.perf_counter() - t_expand:.2f}"
                )
        elif self.debug:
            print("[DEBUG][EXPAND] enabled=false variants=1 rewritten=false seconds=0.00")

        ranked_lists: list[tuple[list[Document], float]] = []
        t_retrieve = time.perf_counter()
        source_lang = detect_language(queries[0])

        for idx, q in enumerate(queries):
            if idx == 0:
                factor = self.original_weight
            elif detect_language(q) == source_lang:
                factor = self.paraphrase_weight
            else:
                factor = self.translation_weight
            ranked_lists.append((self.base_retriever.invoke(q), self.weight_vec * factor))
            ranked_lists.append((self._bm25_docs(q), self.weight_bm25 * factor))
        retrieve_sec = time.perf_counter() - t_retrieve
        t_fusion = time.perf_counter()
        merged = self._weighted_rrf(ranked_lists)[: self.rerank_candidates]
        fusion_sec = time.perf_counter() - t_fusion
        base_docs = [doc for doc, _ in merged]
        if self.debug:
            rerank_mode = "multi_variant" if getattr(self.reranker, "multi_variant_enabled", False) else "single_query"
            print(f"[DEBUG][RERANK] mode={rerank_mode} variants_for_rerank={len(queries) if rerank_mode == 'multi_variant' else 1}")
        t_rerank = time.perf_counter()
        reranked = self.reranker.rerank(
            question,
            base_docs,
            top_n=len(base_docs),
            return_scores=True,
            query_variants=queries,
        )
        rerank_sec = time.perf_counter() - t_rerank
        reranked_scored = [(doc, score) for doc, score, _, _ in reranked]
        top_k = self._dynamic_top_k(reranked_scored, max_k=top_n, ratio=self.dynamic_topk_ratio)
        docs = [doc for doc, _ in reranked_scored[:top_k]]
        self.retrieval_cache[cache_key] = docs
        if self.debug:
            escaped_question = question.replace("`", "'")
            print(f"[DEBUG][RETRIEVE] q={question!r} variants={len(queries)} base_count={len(merged)} rerank_count={len(reranked_scored)} answer_top_k={top_k}")
            for i, q in enumerate(queries, 1):
                print(f"[DEBUG][RETRIEVE][QUERY_VARIANT] {i}={q!r}")
            for i, (doc, score) in enumerate(reranked_scored, 1):
                print(f"[DEBUG][RETRIEVE][RERANK_TOP] rank={i} score={score:.4f} source={_short_source(doc)}")
            print("[DEBUG][RETRIEVE_MD_BEGIN]")
            print("### Retrieval Trace")
            print(f"- Question: `{escaped_question}`")
            print(f"- Base candidates: {len(merged)}")
            print(f"- Reranked kept: {len(reranked_scored)}")
            print("")
            print("#### Base Retrieval")
            print("| Rank | Chunk ID | Score | Source | Type | Chunk Type | Preview |")
            print("| ---: | ---: | ---: | --- | --- | --- | --- |")
            for i, (doc, score) in enumerate(merged, 1):
                print(
                    f"| {i} | {_md_escape(doc.metadata.get('chunk_id', 'unknown'))} | {score:.4f} | "
                    f"{_md_escape(_short_source(doc))} | {_md_escape(doc.metadata.get('source_type', 'unknown'))} | "
                    f"{_md_escape(doc.metadata.get('chunk_type', 'unknown'))} | {_md_escape(_compact_preview(doc.page_content))} |"
                )
            print("")
            print("#### Rerank Result")
            print("| Rank | Chunk ID | Score | Source | Type | Chunk Type | Preview |")
            print("| ---: | ---: | ---: | --- | --- | --- | --- |")
            for i, (doc, score) in enumerate(reranked_scored, 1):
                print(
                    f"| {i} | {_md_escape(doc.metadata.get('chunk_id', 'unknown'))} | {score:.4f} | "
                    f"{_md_escape(_short_source(doc))} | {_md_escape(doc.metadata.get('source_type', 'unknown'))} | "
                    f"{_md_escape(doc.metadata.get('chunk_type', 'unknown'))} | {_md_escape(_compact_preview(doc.page_content))} |"
                )
            print("[DEBUG][RETRIEVE_MD_END]")
            total_sec = time.perf_counter() - t0
            print(
                f"[DEBUG][TIMING][RETRIEVE] expand+routes={retrieve_sec:.2f}s "
                f"fusion={fusion_sec:.2f}s rerank={rerank_sec:.2f}s total={total_sec:.2f}s"
            )
        return docs
