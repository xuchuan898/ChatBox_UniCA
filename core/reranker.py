"""Cross-encoder reranking module."""

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Cross-encoder reranker with optional multi-query variant scoring."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
        alpha: float = 0.5,
        multi_variant_enabled: bool = False,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        self.alpha = alpha
        self.multi_variant_enabled = multi_variant_enabled

    @staticmethod
    def _aggregate_scores(scores_by_variant: list[list[float]]) -> list[float]:
        # Max pooling is robust for multi-query rerank.
        return [max(values) for values in zip(*scores_by_variant)]

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = 8,
        return_scores: bool = False,
        query_variants: list[str] | None = None,
    ):
        """Return reranked docs or detailed scores."""
        if not docs:
            return []
        variants = [query]
        if self.multi_variant_enabled and query_variants:
            # Preserve order and remove duplicates.
            seen = set()
            variants = []
            for q in query_variants:
                key = q.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    variants.append(q.strip())
            if not variants:
                variants = [query]

        scores_by_variant: list[list[float]] = []
        for qv in variants:
            pairs = [(qv, doc.page_content) for doc in docs]
            scores_by_variant.append([float(s) for s in self.model.predict(pairs)])

        final_scores = self._aggregate_scores(scores_by_variant) if len(scores_by_variant) > 1 else scores_by_variant[0]
        scored = []
        for idx, doc in enumerate(docs):
            scored.append(
                (
                    doc,
                    float(final_scores[idx]),
                    float(scores_by_variant[0][idx]),
                    float(final_scores[idx]),
                )
            )
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]
        return top if return_scores else [doc for doc, _, _, _ in top]
