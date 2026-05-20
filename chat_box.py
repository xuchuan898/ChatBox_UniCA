"""Main CLI for local RAG chatbot with persistent index and modular pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from core.config_loader import apply_cli_overrides, load_config
from core.conversation_memory import ConversationMemory
from core.generator import AnswerGenerator
from core.indexer import ensure_index
from core.query_expander import OllamaRewriter, PassthroughRewriter, QueryExpander
from core.reranker import CrossEncoderReranker
from core.retriever import HybridRetriever
from core.semantic_cache import SemanticCache


URL_SOURCE_FILE = "./docs/chroma/source_urls.txt"
DEFAULT_DOC_FILE = "./docs/chroma/master.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbox script.")
    parser.add_argument("--doc-file", type=str, default=DEFAULT_DOC_FILE)
    parser.add_argument("--url-file", type=str, default=None)
    parser.add_argument("-q", "--question", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-chunks-file", type=str, default=None)  # kept for compatibility
    parser.add_argument("--weight-vec", type=float, default=1.25)
    parser.add_argument("--weight-bm25", type=float, default=0.75)
    parser.add_argument("--secondary-variant-weight", type=float, default=0.85)  # compatibility
    parser.add_argument("--variant-mode", type=str, choices=["primary_only", "mapped_current", "mapped_expanded"], default="mapped_current")
    parser.add_argument("--rerank-alpha", type=float, default=0.5)
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--enable-multi-variant-rerank", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--enable-query-expansion", type=str, default=None)
    parser.add_argument("--expansion-model", type=str, default=None)
    parser.add_argument("--expansion-paraphrases", type=int, default=None)
    parser.add_argument("--expansion-add-translation", type=str, default=None)
    parser.add_argument("--expansion-source-lang", type=str, default=None)
    parser.add_argument("--enable-cache", type=str, default=None)
    parser.add_argument("--cache-threshold", type=float, default=None)
    parser.add_argument("--cache-ttl", type=int, default=None)
    parser.add_argument("--enable-memory", type=str, default=None)
    parser.add_argument("--memory-rounds", type=int, default=None)
    return parser.parse_args()


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def _normalize_question_line(line: str) -> str:
    cleaned = re.sub(r"^\s*[-*+]\s*", "", line.strip())
    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned).strip()
    if not cleaned or cleaned.startswith("#") or "?" not in cleaned:
        return ""
    return cleaned


def load_questions_from_file(question_file: str) -> list[str]:
    questions = []
    for raw_line in Path(question_file).read_text(encoding="utf-8").splitlines():
        q = _normalize_question_line(raw_line)
        if q:
            questions.append(q)
    return questions


def prepare_data(doc_file: str, url_file: str | None = None, debug: bool = False, save_chunks_file: str | None = None):
    """Compatibility function used by experiments."""
    _ = save_chunks_file
    state = ensure_index(
        doc_file=doc_file,
        url_file=url_file,
        persist_dir="./index_store",
        embedding_model="intfloat/multilingual-e5-base",
        force_rebuild=False,
        debug=debug,
    )
    return state["vectordb"], state["bm25_index"], state["tokenizer"], state["corpus"]


class _CompatRetriever:
    def __init__(self, retriever: HybridRetriever):
        self.inner = retriever

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.inner.retrieve(query)

    def _get_relevant_documents(self, query: str, run_manager=None) -> list[Document]:
        _ = run_manager
        return self.get_relevant_documents(query)


class _CompatQAChain:
    def __init__(self, retriever: _CompatRetriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def invoke(self, payload: dict[str, Any]) -> dict[str, str]:
        q = payload["input"]
        docs = self.retriever.get_relevant_documents(q)
        return {"answer": self.generator.answer(q, docs)}


def _build_query_expander(config: dict[str, Any]) -> QueryExpander:
    qe_cfg = config["query_expansion"]
    if qe_cfg.get("rewriter") == "passthrough":
        rewriter = PassthroughRewriter()
    else:
        rewriter = OllamaRewriter(model_name=qe_cfg.get("model_name", "gemma3:4b"))
    return QueryExpander(
        rewriter=rewriter,
        num_paraphrases=int(qe_cfg.get("num_paraphrases", 1)),
        add_translation=bool(qe_cfg.get("add_translation", True)),
        source_lang=str(qe_cfg.get("source_lang", "auto")),
        target_lang_for_translation=str(qe_cfg.get("target_lang_for_translation", "auto")),
    )


def chatbox(
    vectordb,
    bm25_index,
    tokenizer,
    corpus,
    debug: bool = False,
    return_prompt: bool = False,
    weight_vec: float = 1.25,
    weight_bm25: float = 0.75,
    secondary_variant_weight: float = 0.85,
    variant_mode: str = "mapped_current",
    rerank_alpha: float = 0.5,
    rerank_candidates: int = 30,
):
    """Compatibility function used by experiments."""
    _ = secondary_variant_weight, variant_mode
    query_expander = QueryExpander(PassthroughRewriter(), num_paraphrases=0, add_translation=False)
    reranker = CrossEncoderReranker(alpha=rerank_alpha, multi_variant_enabled=False)
    retriever = HybridRetriever(
        vectordb=vectordb,
        bm25_index=bm25_index,
        tokenizer=tokenizer,
        corpus=corpus,
        reranker=reranker,
        weight_vec=weight_vec,
        weight_bm25=weight_bm25,
        rerank_candidates=rerank_candidates,
        query_expander=query_expander,
        query_expansion_enabled=False,
        debug=debug,
    )
    compat_retriever = _CompatRetriever(retriever)
    generator = AnswerGenerator()
    qa_chain = _CompatQAChain(compat_retriever, generator)
    if return_prompt:
        return qa_chain, compat_retriever, None
    return qa_chain, compat_retriever


def _serialize_docs(docs: list[Document]) -> list[dict[str, Any]]:
    return [{"chunk_id": d.metadata.get("chunk_id"), "source": d.metadata.get("source"), "preview": d.page_content[:220]} for d in docs]


def _write_chunks_jsonl(path: str, corpus: list[Document]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for idx, doc in enumerate(corpus):
            handle.write(
                json.dumps(
                    {
                        "chunk_id": doc.metadata.get("chunk_id", idx),
                        "content": doc.page_content,
                        "chunk_type": doc.metadata.get("chunk_type", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "length": len(doc.page_content),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    os.environ["USER_AGENT"] = "MyChatBot/1.0"
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "indexing.force_rebuild": args.force_rebuild if args.force_rebuild else None,
        "retrieval.weight_vec": args.weight_vec,
        "retrieval.weight_bm25": args.weight_bm25,
        "retrieval.rerank_alpha": args.rerank_alpha,
        "retrieval.rerank_candidates": args.rerank_candidates,
        "retrieval.multi_variant_rerank_enabled": _parse_bool(args.enable_multi_variant_rerank),
        "query_expansion.enabled": _parse_bool(args.enable_query_expansion),
        "query_expansion.model_name": args.expansion_model,
        "query_expansion.num_paraphrases": args.expansion_paraphrases,
        "query_expansion.add_translation": _parse_bool(args.expansion_add_translation),
        "query_expansion.source_lang": args.expansion_source_lang,
        "cache.enabled": _parse_bool(args.enable_cache),
        "cache.threshold": args.cache_threshold,
        "cache.ttl": args.cache_ttl,
        "memory.enabled": _parse_bool(args.enable_memory),
        "memory.short_term_rounds": args.memory_rounds,
    }
    config = apply_cli_overrides(config, overrides)

    state = ensure_index(
        doc_file=args.doc_file,
        url_file=args.url_file,
        persist_dir=config["indexing"]["persist_dir"],
        embedding_model="intfloat/multilingual-e5-base",
        force_rebuild=bool(config["indexing"]["force_rebuild"]),
        debug=args.debug,
    )
    if args.save_chunks_file:
        _write_chunks_jsonl(args.save_chunks_file, state["corpus"])
        if args.debug:
            print(f"[DEBUG][CHUNKS] saved={args.save_chunks_file} count={len(state['corpus'])}")
    if args.debug:
        meta = state.get("meta", {})
        print(
            f"[DEBUG][BOOT] doc={args.doc_file} chunks={meta.get('chunk_count', 'unknown')} "
            f"embedding={meta.get('embedding_model', 'intfloat/multilingual-e5-base')}"
        )
    query_expander = _build_query_expander(config)
    if args.debug:
        qe = config["query_expansion"]
        print(
            f"[DEBUG][BOOT] query_expansion enabled={qe['enabled']} rewriter={qe['rewriter']} "
            f"model={qe['model_name']} paraphrases={qe['num_paraphrases']} "
            f"add_translation={qe['add_translation']} source_lang={qe['source_lang']} "
            f"target_lang={qe['target_lang_for_translation']}"
        )
        print(
            f"[DEBUG][BOOT] rerank multi_variant_enabled="
            f"{bool(config['retrieval'].get('multi_variant_rerank_enabled', False))} "
            f"alpha={config['retrieval']['rerank_alpha']}"
        )
    reranker = CrossEncoderReranker(
        alpha=float(config["retrieval"]["rerank_alpha"]),
        multi_variant_enabled=bool(config["retrieval"].get("multi_variant_rerank_enabled", False)),
    )
    retriever = HybridRetriever(
        vectordb=state["vectordb"],
        bm25_index=state["bm25_index"],
        tokenizer=state["tokenizer"],
        corpus=state["corpus"],
        reranker=reranker,
        weight_vec=float(config["retrieval"]["weight_vec"]),
        weight_bm25=float(config["retrieval"]["weight_bm25"]),
        rerank_candidates=int(config["retrieval"]["rerank_candidates"]),
        query_expander=query_expander,
        query_expansion_enabled=bool(config["query_expansion"]["enabled"]),
        debug=args.debug,
    )
    generator = AnswerGenerator(
        model_name=config["generation"]["model_name"],
        temperature=float(config["generation"]["temperature"]),
        num_predict=int(config["generation"]["num_predict"]),
    )
    cache = None
    if config["cache"]["enabled"]:
        cache = SemanticCache(threshold=float(config["cache"]["threshold"]), ttl=int(config["cache"]["ttl"]), debug=args.debug)
    if args.debug:
        print(
            f"[DEBUG][BOOT] cache enabled={bool(cache)} "
            f"threshold={config['cache']['threshold']} ttl={config['cache']['ttl']}"
        )
    memory = None
    if config["memory"]["enabled"]:
        memory = ConversationMemory(rounds=int(config["memory"]["short_term_rounds"]), model_name=config["generation"]["model_name"])
    if args.debug:
        print(
            f"[DEBUG][BOOT] memory enabled={bool(memory)} rounds={config['memory']['short_term_rounds']}"
        )

    def run_one(question: str) -> str:
        t_total = time.perf_counter()
        mem_ctx = ""
        t_mem = time.perf_counter()
        if memory:
            mem_ctx = "\n".join(filter(None, [memory.build_short_prompt(), memory.build_long_prompt()]))
        mem_sec = time.perf_counter() - t_mem
        t_cache = time.perf_counter()
        if cache:
            hit = cache.get(question)
            if hit:
                if args.debug:
                    print(f"[DEBUG][CACHE] served=true seconds={time.perf_counter() - t_cache:.2f}")
                    print(f"[DEBUG][TIMING][RUN_ONE] memory={mem_sec:.2f}s total={time.perf_counter() - t_total:.2f}s")
                return str(hit.get("answer", ""))
        if args.debug:
            print(f"[DEBUG][CACHE] served=false seconds={time.perf_counter() - t_cache:.2f}")
        t_retrieve = time.perf_counter()
        docs = retriever.retrieve(question)
        retrieve_sec = time.perf_counter() - t_retrieve
        t_prompt = time.perf_counter()
        prompt = generator.render_prompt(question, docs, mem_ctx)
        prompt_sec = time.perf_counter() - t_prompt
        if args.debug:
            print("[DEBUG][PROMPT_MD_BEGIN]")
            print("```text")
            print(prompt)
            print("```")
            print("[DEBUG][PROMPT_MD_END]")
        t_gen = time.perf_counter()
        answer = generator.answer(question, docs, mem_ctx)
        gen_sec = time.perf_counter() - t_gen
        t_write = time.perf_counter()
        if cache:
            cache.put(question, answer, _serialize_docs(docs))
        if memory:
            memory.add_turn(question, answer)
        write_sec = time.perf_counter() - t_write
        if args.debug:
            print(
                f"[DEBUG][TIMING][RUN_ONE] memory={mem_sec:.2f}s retrieve={retrieve_sec:.2f}s "
                f"render_prompt={prompt_sec:.2f}s generate={gen_sec:.2f}s "
                f"persist(cache+memory)={write_sec:.2f}s total={time.perf_counter() - t_total:.2f}s"
            )
        return answer

    if args.question_file:
        questions = load_questions_from_file(args.question_file)
        lines = []
        for idx, q in enumerate(questions, 1):
            start = time.perf_counter()
            ans = run_one(q)
            elapsed = time.perf_counter() - start
            print(f"[Q{idx}] {q}")
            print(f"[A{idx}] {ans}")
            if args.debug:
                print(f"[DEBUG] QA invoke time: {elapsed:.2f}s")
            lines.extend([f"[Q{idx}] {q}", f"[A{idx}] {ans}", ""])
        if args.answer_file:
            Path(args.answer_file).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return

    if args.question:
        print(f"**Bot Answer**: {run_one(args.question)}")
        return

    print("Chat started. Type 'exit' to quit.")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if user_input:
            print(f"**Bot Answer**: {run_one(user_input)}")


if __name__ == "__main__":
    main()
