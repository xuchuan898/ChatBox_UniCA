"""Lightweight evaluation helpers."""

from __future__ import annotations

from typing import Iterable


def context_recall(retrieved_chunk_ids: Iterable[int], gold_chunk_ids: Iterable[int]) -> float:
    """Compute Context Recall against gold chunk ids."""
    retrieved = set(int(x) for x in retrieved_chunk_ids)
    gold = set(int(x) for x in gold_chunk_ids)
    if not gold:
        return 0.0
    return len(retrieved & gold) / len(gold)


def ragas_placeholder() -> dict:
    """Reserved hook for future RAGAS integration."""
    return {"status": "not_implemented"}
