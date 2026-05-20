"""Query expansion utilities for multilingual RAG retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import List

from langchain_ollama import ChatOllama


def detect_language(text: str) -> str:
    """Detect query language. Returns 'en', 'fr', or 'unknown'."""
    lower = f" {text.lower()} "
    fr_markers = [
        " le ", " la ", " les ", " des ", " du ", " un ", " une ",
        " dans ", " avec ", " pour ", " est ", " responsable ", " durée ", " courriel ",
    ]
    en_markers = [
        " the ", " and ", " for ", " with ", " what ", " which ", " is ", " are ",
        " duration ", " internship ", " email ", " responsible ",
    ]
    fr_score = sum(marker in lower for marker in fr_markers)
    en_score = sum(marker in lower for marker in en_markers)
    if re.search(r"[àâçéèêëîïôûùüÿœ]", lower):
        fr_score += 2
    if fr_score == 0 and en_score == 0:
        return "unknown"
    return "fr" if fr_score >= en_score + 1 else "en"


class BaseQueryRewriter(ABC):
    """Abstract query rewriter interface."""

    @abstractmethod
    def rewrite(self, query: str, lang: str, num_variants: int) -> list[str]:
        """Return same-language paraphrases."""

    @abstractmethod
    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate query into target language."""


class PassthroughRewriter(BaseQueryRewriter):
    """Baseline no-op rewriter for ablations."""

    def rewrite(self, query: str, lang: str, num_variants: int) -> list[str]:
        return []

    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        return query


class OllamaRewriter(BaseQueryRewriter):
    """LLM-backed query rewriter using Ollama chat models."""

    def __init__(self, model_name: str = "gemma3:4b", temperature: float = 0.0):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            seed=42,
            validate_model_on_init=True,
            num_predict=256,
        )

    def rewrite(self, query: str, lang: str, num_variants: int) -> list[str]:
        if num_variants <= 0:
            return []
        prompt = (
            "You rewrite user search queries for retrieval.\n"
            f"Language: {lang}\n"
            f"Generate exactly {num_variants} paraphrases in same language.\n"
            "Rules: keep meaning, concise, no explanations, one per line.\n"
            f"Query: {query}"
        )
        text = self.llm.invoke(prompt).content
        lines = [line.strip(" -\t") for line in str(text).splitlines() if line.strip()]
        unique = []
        seen = {query.lower()}
        for line in lines:
            if line.lower() not in seen:
                seen.add(line.lower())
                unique.append(line)
            if len(unique) >= num_variants:
                break
        return unique

    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return query
        prompt = (
            "Translate this search query for retrieval.\n"
            f"From: {source_lang} To: {target_lang}\n"
            "Return only translated query, no quotes, no explanations.\n"
            f"Query: {query}"
        )
        text = str(self.llm.invoke(prompt).content).strip()
        return text or query


class QueryExpander:
    """Expand user query with paraphrases and cross-language translation."""

    def __init__(
        self,
        rewriter: BaseQueryRewriter,
        num_paraphrases: int = 1,
        add_translation: bool = True,
        source_lang: str = "auto",
        target_lang_for_translation: str = "auto",
    ):
        self.rewriter = rewriter
        self.num_paraphrases = max(0, int(num_paraphrases))
        self.add_translation = add_translation
        self.source_lang = source_lang
        self.target_lang_for_translation = target_lang_for_translation

    def _resolve_source_lang(self, query: str) -> str:
        if self.source_lang in {"en", "fr"}:
            return self.source_lang
        detected = detect_language(query)
        return detected if detected in {"en", "fr"} else "en"

    def _resolve_target_lang(self, source_lang: str) -> str:
        if self.target_lang_for_translation in {"en", "fr"}:
            return self.target_lang_for_translation
        return "fr" if source_lang == "en" else "en"

    def expand(self, query: str) -> list[str]:
        """Expand query. Fallback to original query on any failure."""
        base = " ".join(query.split())
        if not base:
            return [query]
        try:
            source_lang = self._resolve_source_lang(base)
            result: List[str] = [base]
            if self.num_paraphrases > 0:
                result.extend(self.rewriter.rewrite(base, source_lang, self.num_paraphrases))
            if self.add_translation:
                target = self._resolve_target_lang(source_lang)
                result.append(self.rewriter.translate(base, source_lang, target))
            dedup = []
            seen = set()
            for item in result:
                cleaned = " ".join(str(item).split())
                key = cleaned.lower()
                if cleaned and key not in seen:
                    seen.add(key)
                    dedup.append(cleaned)
            return dedup or [base]
        except Exception:
            return [base]
