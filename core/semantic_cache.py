"""Semantic cache with cosine-similarity lookup."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """JSON-backed semantic cache."""

    def __init__(
        self,
        path: str | Path = "./index_store/semantic_cache.json",
        embedding_model: str = "intfloat/multilingual-e5-base",
        threshold: float = 0.92,
        ttl: int = 86400,
        debug: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.threshold = threshold
        self.ttl = ttl
        self.debug = debug
        self.items = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def get(self, query: str) -> dict[str, Any] | None:
        q_emb = self.embedder.embed_query(query)
        now = int(time.time())
        best = None
        best_sim = -1.0
        alive = []
        for item in self.items:
            if now - int(item.get("timestamp", 0)) <= self.ttl:
                alive.append(item)
                sim = _cosine(q_emb, item.get("embedding", []))
                if sim > best_sim:
                    best_sim = sim
                    best = item
        self.items = alive
        if self.debug:
            print(f"[DEBUG][CACHE] candidates={len(alive)} best_similarity={best_sim:.4f}")
        if best and best_sim >= self.threshold:
            if self.debug:
                print(f"[DEBUG][CACHE] HIT threshold={self.threshold:.2f}")
            return best
        return None

    def put(self, query: str, answer: str, context_docs: list[dict[str, Any]]) -> None:
        self.items.append({
            "query": query,
            "answer": answer,
            "context_docs": context_docs,
            "timestamp": int(time.time()),
            "embedding": self.embedder.embed_query(query),
        })
        self._save()
