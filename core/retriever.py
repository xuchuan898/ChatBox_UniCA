"""Hybrid retrieval (vector + BM25 + RRF) with query expansion support."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any

from langchain_core.documents import Document

from .query_expander import QueryExpander
from .reranker import CrossEncoderReranker


def _short_source(doc: Document) -> str:
    src = str(doc.metadata.get("source", "unknown"))
    return src.split("/")[-1].split("\\")[-1]


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
        rerank_candidates: int = 30,
        query_expander: QueryExpander | None = None,
        query_expansion_enabled: bool = False,
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
        self.rerank_candidates = rerank_candidates
        self.query_expander = query_expander
        self.query_expansion_enabled = query_expansion_enabled
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
                doc_id = int(doc.metadata.get("chunk_id", hash(doc.page_content)))
                scores[doc_id] += weight / (k + rank)
                by_id.setdefault(doc_id, doc)
        return sorted(((by_id[key], val) for key, val in scores.items()), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _dynamic_top_k(reranked_scored: list[tuple[Document, float]]) -> int:
        if not reranked_scored:
            return 0
        base = min(8, len(reranked_scored))
        cutoff = reranked_scored[base - 1][1]
        tail = [s for _, s in reranked_scored[base : min(len(reranked_scored), base + 6)]]
        close = sum(1 for s in tail if (cutoff - s) <= 0.03)
        return min(len(reranked_scored), base + close)

    def retrieve(self, question: str) -> list[Document]:
        t0 = time.perf_counter()
        if question in self.retrieval_cache:
            if self.debug:
                print(f"[DEBUG][RETRIEVE][CACHE_HIT] q={question!r}")
            return self.retrieval_cache.pop(question)
        queries = [question]
        if self.query_expansion_enabled and self.query_expander:
            t_expand = time.perf_counter()
            queries = self.query_expander.expand(question)
            if self.debug:
                print(f"[DEBUG][EXPAND] enabled=true variants={len(queries)} seconds={time.perf_counter() - t_expand:.2f}")
        elif self.debug:
            print("[DEBUG][EXPAND] enabled=false variants=1 seconds=0.00")
        ranked_lists: list[tuple[list[Document], float]] = []
        t_retrieve = time.perf_counter()
        for idx, q in enumerate(queries):
            factor = 1.0 if idx == 0 else 0.85
            ranked_lists.append((self.base_retriever.invoke(q), self.weight_vec * factor))
            ranked_lists.append((self._bm25_docs(q), self.weight_bm25 * factor))
        retrieve_sec = time.perf_counter() - t_retrieve
        t_fusion = time.perf_counter()
        merged = self._weighted_rrf(ranked_lists)[: self.rerank_candidates]
        fusion_sec = time.perf_counter() - t_fusion
        base_docs = [doc for doc, _ in merged]
        if self.debug and hasattr(self.reranker, "_translated_query"):
            translated = self.reranker._translated_query(question)
            print(f"[DEBUG][RERANK_DUAL][QUERY] src={question!r} translated={translated!r} alpha={self.reranker.alpha:.3f}")
        t_rerank = time.perf_counter()
        reranked = self.reranker.rerank(question, base_docs, top_n=len(base_docs), return_scores=True)
        rerank_sec = time.perf_counter() - t_rerank
        reranked_scored = [(doc, score) for doc, score, _, _ in reranked]
        top_k = self._dynamic_top_k(reranked_scored)
        docs = [doc for doc, _ in reranked_scored[:top_k]]
        self.retrieval_cache[question] = docs
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
            print("| Rank | Chunk ID | Score | Source |")
            print("| ---: | ---: | ---: | --- |")
            for i, (doc, score) in enumerate(merged, 1):
                print(f"| {i} | {doc.metadata.get('chunk_id', 'unknown')} | {score:.4f} | {_short_source(doc)} |")
            print("")
            print("#### Rerank Result")
            print("| Rank | Chunk ID | Score | Source |")
            print("| ---: | ---: | ---: | --- |")
            for i, (doc, score) in enumerate(reranked_scored, 1):
                print(f"| {i} | {doc.metadata.get('chunk_id', 'unknown')} | {score:.4f} | {_short_source(doc)} |")
            print("[DEBUG][RETRIEVE_MD_END]")
            total_sec = time.perf_counter() - t0
            print(
                f"[DEBUG][TIMING][RETRIEVE] expand+routes={retrieve_sec:.2f}s "
                f"fusion={fusion_sec:.2f}s rerank={rerank_sec:.2f}s total={total_sec:.2f}s"
            )
        return docs
