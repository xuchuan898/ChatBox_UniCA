"""Load config.yaml and merge with CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "indexing": {"persist_dir": "./index_store", "force_rebuild": False},
    "retrieval": {
        "weight_vec": 1.25,
        "weight_bm25": 0.75,
        "original_weight": 1.0,
        "paraphrase_weight": 1.2,
        "translation_weight": 0.85,
        "rerank_candidates": 30,
        "rerank_alpha": 0.5,
        "multi_variant_rerank_enabled": False,
        "disable_rerank": False,
    },
    "query_expansion": {
        "enabled": True,
        "rewriter": "ollama",
        "model_name": "gemma3:4b",
        "num_paraphrases": 1,
        "add_translation": True,
        "source_lang": "auto",
        "target_lang_for_translation": "auto",
        "multi_turn": {"enabled": True, "max_history_turns": 5},
    },
    "cache": {"enabled": True, "threshold": 0.92, "ttl": 86400},
    "memory": {"enabled": True, "short_term_rounds": 5, "long_term_enabled": False},
    "generation": {"model_name": "gemma3:4b", "temperature": 0.0, "num_predict": 256},
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load config with defaults."""
    path = Path(config_path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def apply_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides to config."""
    out = copy.deepcopy(config)
    for dotted, value in overrides.items():
        if value is None:
            continue
        keys = dotted.split(".")
        cursor = out
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return out
