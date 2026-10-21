"""Provider selection for the CLI and API, without changing the core pipeline."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AlbertClient:
    """Small Albert HTTP client. Credentials come from the environment or a private local file."""

    def __init__(self, config: dict):
        cfg = config.get("albert", {})
        self.api_key = os.environ.get(cfg.get("api_key_env", "ALBERT_API_KEY"), "").strip()
        key_file = cfg.get("api_key_file")
        if not self.api_key and key_file:
            path = Path(key_file).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            try:
                self.api_key = path.read_text(encoding="utf-8-sig").strip()
            except OSError:
                raise ValueError("Cannot read the configured Albert API key file") from None
        if not self.api_key:
            raise ValueError(f"Set {cfg.get('api_key_env', 'ALBERT_API_KEY')} or configure albert.api_key_file to use Albert API")
        base = os.environ.get("ALBERT_BASE_URL") or cfg.get("base_url", "https://albert.api.etalab.gouv.fr/v1")
        self.base_url = base.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout = float(cfg.get("timeout", 120))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("albert.timeout must be a positive finite number")

    def post(self, endpoint: str, payload: dict) -> dict:
        req = Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            hints = {401: "check the API key", 403: "check account access", 404: "check the model ID and base URL",
                     429: "rate limit or quota exceeded", 503: "model temporarily unavailable"}
            # Do not include response bodies or request headers in public errors.
            raise RuntimeError(f"Albert {endpoint} failed (HTTP {exc.code}): {hints.get(exc.code, 'request rejected')}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError(f"Albert {endpoint} could not be reached or timed out") from None
        except (ValueError, UnicodeError):
            raise RuntimeError(f"Albert {endpoint} returned invalid JSON") from None
        if not isinstance(result, dict):
            raise RuntimeError(f"Albert {endpoint} returned an invalid response")
        return result


class AlbertChat:
    """invoke interface shared by answer generation and query rewriting."""

    def __init__(self, client: AlbertClient, model_name: str, temperature: float = 0.0, num_predict: int = 256):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict

    def invoke(self, prompt: str):
        result = self.client.post("chat/completions", {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.num_predict,
            "stream": False,
        })
        try:
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Albert chat/completions returned no text answer") from None
        return SimpleNamespace(content=content)


class AlbertReranker:
    """Albert rerank adapter preserving core rerank result tuples and max pooling."""

    def __init__(self, client: AlbertClient, model_name: str, alpha: float = 0.5, multi_variant_enabled: bool = False):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.client = client
        self.model_name = model_name
        self.alpha = alpha
        self.multi_variant_enabled = multi_variant_enabled

    def _score(self, query, docs):
        result = self.client.post("rerank", {
            "model": self.model_name, "query": query,
            "documents": [doc.page_content for doc in docs], "top_n": len(docs),
        })
        scores = [None] * len(docs)
        try:
            for item in result["results"]:
                idx = item["index"]
                if type(idx) is not int or not 0 <= idx < len(docs) or scores[idx] is not None:
                    raise ValueError()
                score = float(item["relevance_score"])
                if not math.isfinite(score):
                    raise ValueError()
                scores[idx] = score
            if any(score is None for score in scores):
                raise ValueError()
        except (KeyError, TypeError, ValueError, OverflowError):
            raise RuntimeError("Albert rerank returned invalid or incomplete document scores") from None
        return scores

    def rerank(self, query, docs, top_n=8, return_scores=False, query_variants=None):
        if not docs or top_n <= 0:
            return []
        variants = [query]
        if self.multi_variant_enabled and query_variants:
            variants = []
            seen = set()
            for variant in query_variants:
                key = variant.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    variants.append(variant.strip())
            variants = variants or [query]
        scores = [self._score(variant, docs) for variant in variants]
        final = [max(values) for values in zip(*scores)]
        ranked = [(doc, final[i], scores[0][i], final[i]) for i, doc in enumerate(docs)]
        ranked.sort(key=lambda row: row[1], reverse=True)
        return ranked[:top_n] if return_scores else [row[0] for row in ranked[:top_n]]


def build_generator(config):
    from core.generator import AnswerGenerator
    cfg = config.get("generation", {})
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return AnswerGenerator(model_name=cfg.get("model_name", "gemma3:4b"),
                               temperature=cfg.get("temperature", 0.0), num_predict=cfg.get("num_predict", 256))
    if provider != "albert":
        raise ValueError(f"Unknown generation provider: {provider}")

    class AlbertAnswerGenerator(AnswerGenerator):
        def __init__(self):
            self.llm = AlbertChat(AlbertClient(config), cfg.get("albert_model_name", "openweight-medium"),
                                  cfg.get("temperature", 0.0), cfg.get("num_predict", 256))

    return AlbertAnswerGenerator()


def build_reranker(config):
    cfg = config.get("retrieval", {})
    provider = cfg.get("rerank_provider", "local")
    kwargs = {"alpha": cfg.get("rerank_alpha", 0.5),
              "multi_variant_enabled": cfg.get("multi_variant_rerank_enabled", False)}
    if provider == "albert":
        return AlbertReranker(AlbertClient(config), cfg.get("albert_rerank_model_name", "openweight-rerank"), **kwargs)
    if provider != "local":
        raise ValueError(f"Unknown rerank provider: {provider}")
    from core.reranker import CrossEncoderReranker
    return CrossEncoderReranker(model_name=cfg.get("rerank_model_name", "nvidia/llama-nemotron-rerank-1b-v2"), **kwargs)


def build_query_expander(config):
    from core.query_expander import OllamaRewriter, PassthroughRewriter, QueryExpander, QwenRewriter
    cfg = config.get("query_expansion", {})
    provider = cfg.get("rewriter", "ollama")
    if not cfg.get("enabled", True) or provider == "passthrough":
        rewriter = PassthroughRewriter()
    elif provider == "ollama":
        rewriter = OllamaRewriter(model_name=cfg.get("model_name", "gemma3:4b"))
    elif provider == "albert":
        class AlbertRewriter(OllamaRewriter):
            def __init__(self):
                self.llm = AlbertChat(AlbertClient(config), cfg.get("albert_model_name", "openweight-medium"))
        rewriter = AlbertRewriter()
    elif provider == "qwen":
        rewriter = QwenRewriter()
    else:
        raise ValueError(f"Unknown query rewriter: {provider}")
    return QueryExpander(rewriter=rewriter, num_paraphrases=cfg.get("num_paraphrases", 1),
                         add_translation=cfg.get("add_translation", True), source_lang=cfg.get("source_lang", "auto"),
                         target_lang_for_translation=cfg.get("target_lang_for_translation", "auto"))


def needs_ollama(config):
    return (config.get("generation", {}).get("provider", "ollama") == "ollama" or
            (config.get("query_expansion", {}).get("enabled", True) and
             config.get("query_expansion", {}).get("rewriter", "ollama") == "ollama"))
