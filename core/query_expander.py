"""Query expansion utilities for multilingual RAG retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional

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
    def rewrite(
        self,
        query: str,
        lang: str,
        num_variants: int,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> list[str]:
        """Return same-language paraphrases."""

    @abstractmethod
    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate query into target language."""


class PassthroughRewriter(BaseQueryRewriter):
    """Baseline no-op rewriter for ablations."""

    def rewrite(
        self,
        query: str,
        lang: str,
        num_variants: int,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> list[str]:
        _ = query, lang, num_variants, history
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

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        lines = []
        role_labels = {"user": "User", "assistant": "Assistant"}
        for item in history:
            role = role_labels.get(str(item.get("role", "")).lower())
            content = " ".join(str(item.get("content", "")).split())
            if role and content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def rewrite(
        self,
        query: str,
        lang: str,
        num_variants: int,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> list[str]:
        if num_variants <= 0:
            return []
        formatted_history = self._format_history(history or [])
        if formatted_history:
            REWRITE_PROMPT_WITH_HISTORY = (
                "You are a query rewriter for a retrieval system. \n"
                "Given the conversation history, rewrite the user's last question into a standalone query that can be understood without context.\n"
                "\n"
                "You MUST follow these steps:\n"
                "1. Identify any vague references in the current question (pronouns like \"it\", \"they\", \"this\", \"that\", \"him\", or omitted subjects/objects).\n"
                "2. Look back at the history to find the exact entity or phrase being referred to.\n"
                "3. Rewrite the question by replacing the vague reference with that entity, and restore any missing details.\n"
                "4. If the question is already standalone, return it unchanged.\n"
                "\n"
                "Output ONLY the final rewritten query, no explanation.\n"
                "\n"
                "Example 1:\n"
                "History:\n"
                "User: When was the Eiffel Tower built?\n"
                "Assistant: It was completed in 1889.\n"
                "Current question: How tall is it?\n"
                "Rewritten query: How tall is the Eiffel Tower?\n"
                "\n"
                "Example 2:\n"
                "History:\n"
                "User: Show me Italian restaurants nearby.\n"
                "Assistant: There are three: La Piazza, Bella Napoli, and Il Forno.\n"
                "Current question: What about the first one?\n"
                "Rewritten query: What about La Piazza?\n"
                "\n"
                "Now process the following:\n"
                "History:\n"
                f"{formatted_history}\n"
                f"Current question: {query}\n"
                "Rewritten query:"
            )
            prompt = REWRITE_PROMPT_WITH_HISTORY
        else:
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


class QwenRewriter(BaseQueryRewriter):
    """Qwen2.5-based query rewriter for multi-turn context rewriting."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required for QwenRewriter: pip install transformers")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

    @staticmethod
    def _format_messages(query: str, history: Optional[List[Dict[str, str]]]) -> list[dict]:
        system_prompt = (
            "You are a query rewriter for a retrieval system. "
            "Given conversation history, rewrite the user's last question into a standalone query. "
            "Resolve all pronouns (e.g., 'it', 'this', 'that') and restore omitted details. "
            "Output ONLY the final rewritten query on a single line, without any additional text.\n\n"
            "Examples:\n"
            "History:\nUser: What courses does Prof. Smith teach?\nAssistant: He teaches NLP and Data Mining.\n"
            "Current question: Which one is in S1?\nRewritten query: Which course is in S1, NLP or Data Mining?\n\n"
            "History:\nUser: How many ECTS is the internship?\nAssistant: It is 6 ECTS.\n"
            "Current question: And the duration?\nRewritten query: What is the duration of the internship?"
        )
        msgs = [{"role": "system", "content": system_prompt}]

        if history:
            hist_lines = []
            for item in history[-5:]:
                role = item.get("role", "").capitalize()
                content = " ".join(str(item.get("content", "")).split())
                if content:
                    hist_lines.append(f"{role}: {content}")
            hist_text = "\n".join(hist_lines)
            user_content = f"History:\n{hist_text}\n\nCurrent question: {query}"
        else:
            user_content = f"Rewrite to standalone: {query}"

        msgs.append({"role": "user", "content": user_content})
        return msgs

    def rewrite(
            self,
            query: str,
            lang: str,
            num_variants: int,
            history: Optional[List[Dict[str, str]]] = None,
    ) -> list[str]:
        _ = lang
        if num_variants <= 0:
            return []

        try:
            messages = self._format_messages(query, history)
            # 关键修复：add_generation_prompt=True 会在末尾加 <|im_start|>assistant\n
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("[DEBUG][QWEN_PROMPT]", prompt)  # 可保留调试

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # 只取模型生成的部分，去掉输入 prompt
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print("[DEBUG][QWEN_RAW_OUTPUT]", raw)

            # 清理可能残留的 "assistant:" 前缀和特殊 token
            if raw.lower().startswith("assistant:"):
                raw = raw[len("assistant:"):].strip()
            import re
            raw = re.sub(r'<\|.*?\|>', '', raw).strip()

            # 取第一行作为改写结果
            rewritten = raw.split("\n")[0].strip().strip('"').strip("'")

            # 去重：如果和原问题完全一样或为空，返回空列表
            if not rewritten or rewritten.lower() == query.lower():
                return []
            return [rewritten]

        except Exception:
            return []

    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        """QwenRewriter does not support translation. Returns query unchanged.
        TODO: migrate to a dedicated translation model or reuse OllamaRewriter.translate."""
        _ = source_lang, target_lang
        return query


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

    @staticmethod
    def _is_rewrite_valid(original: str, rewritten: str) -> bool:
        """Validate rewrite quality. Returns False if rewrite is useless or still contains unresolved pronouns."""
        rw = " ".join(rewritten.split())
        orig = " ".join(original.split())
        if not rw:
            return False
        # Reject if raw model returned original verbatim
        if rw.lower() == orig.lower():
            return False
        # Reject if too similar (less than 3 chars added or 2+ chars different)
        diff_len = len(rw) - len(orig)
        if diff_len < 3 and rw.lower().startswith(orig.lower()[:10]):
            return False
        # Reject if unresolved deictic pronouns remain
        unresolved = re.findall(r"\b(it|they|this|that|these|those|its|their|them)\b", rw.lower())
        if unresolved:
            return False
        # Reject if rewritten introduces a named entity absent from original in cases
        # where original has no proper name and rewritten gratuitously adds one.
        orig_proper = re.findall(r"\b[A-Z][a-zà-ü]+\b", orig)
        rw_proper = re.findall(r"\b[A-Z][a-zà-ü]+\b", rw)
        if not orig_proper and rw_proper:
            for name in rw_proper:
                if name.lower() not in orig.lower():
                    return False
        return True

    def expand(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> list[str]:
        """Expand query. Fallback to original query on any failure."""
        base = " ".join(query.split())
        if not base:
            return [query]
        try:
            source_lang = self._resolve_source_lang(base)
            result: List[str] = [base]
            if self.num_paraphrases > 0:
                rewrites = self.rewriter.rewrite(base, source_lang, self.num_paraphrases, history=history)
                for rw in rewrites:
                    if self._is_rewrite_valid(base, rw):
                        result.append(rw)
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