"""Cross-encoder reranking module."""

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .query_expander import detect_language


class CrossEncoderReranker:
    """Dual-query reranker with translated-query fusion."""

    def __init__(self, model_name: str = "nvidia/llama-nemotron-rerank-1b-v2", alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        self.alpha = alpha

    def _translated_query(self, query: str) -> str:
        lang = detect_language(query)
        mapping = {
            "en": {"work-study": "alternance", "responsible": "responsable", "email": "courriel", "internship": "stage"},
            "fr": {"alternance": "work-study", "responsable": "responsible", "courriel": "email", "stage": "internship"},
        }
        output = query
        for src, dst in mapping.get(lang, {}).items():
            output = output.replace(src, dst)
        return output

    def rerank(self, query: str, docs: list[Document], top_n: int = 8, return_scores: bool = False):
        """Return reranked docs or detailed scores."""
        if not docs:
            return []
        src_scores = [float(s) for s in self.model.predict([(query, doc.page_content) for doc in docs])]
        translated = self._translated_query(query)
        trans_scores = src_scores if self.alpha == 1.0 else [float(s) for s in self.model.predict([(translated, doc.page_content) for doc in docs])]
        scored = []
        for doc, s1, s2 in zip(docs, src_scores, trans_scores):
            score = self.alpha * s1 + (1.0 - self.alpha) * s2
            scored.append((doc, float(score), float(s1), float(s2)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]
        return top if return_scores else [doc for doc, _, _, _ in top]
