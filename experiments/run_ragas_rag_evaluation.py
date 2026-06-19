from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_loader import load_config
from core.conversation_memory import ConversationMemory
from core.generator import AnswerGenerator
from core.indexer import ensure_index
from core.query_expander import OllamaRewriter, PassthroughRewriter, QueryExpander, QwenRewriter
from core.reranker import CrossEncoderReranker
from core.retriever import HybridRetriever


DEFAULT_RAGAS_EXPERIMENT = {
    "dataset_path": "questions/rag_memory_challenge_dataset.json",
    "output_dir": "experiments/results/ragas_memory_eval",
    "results_file": "evaluation_results.json",
    "log_file": "experiment.log",
    "doc_file": "docs/chroma/master.md",
    "ollama_host": "http://127.0.0.1:11434",
    "ollama_bin_dir": "~/ollama/bin",
    "ollama_log_file": "~/ollama/ollama.log",
    "auto_start_ollama": False,
    "evaluation_llm_model": "qwen2.5:32b",
    "evaluation_llm_temperature": 0.0,
    "evaluation_embedding_model": "nomic-embed-text",
    "generation_model": "qwen2.5:32b",
    "generation_temperature": 0.0,
    "generation_num_predict": 256,
    "rerank_candidates": 10,
    "ragas_timeout_seconds": 300,
    "ragas_max_workers": 1,
    "debug_retrieval": False,
    "ragas_raise_exceptions": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn RAG with Ragas and custom retrieval metrics.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML config path.")
    return parser.parse_args()


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ragas_rag_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def ollama_is_ready(ollama_host: str) -> bool:
    try:
        with urlopen(f"{ollama_host.rstrip('/')}/api/tags", timeout=2) as _:
            return True
    except (URLError, TimeoutError, OSError):
        return False


def ensure_ollama(exp_cfg: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    ollama_host = str(exp_cfg["ollama_host"])
    ollama_bin_dir = Path(str(exp_cfg["ollama_bin_dir"])).expanduser().resolve()
    ollama_bin = ollama_bin_dir / "ollama"
    if os.name == "nt" and not ollama_bin.exists():
        ollama_bin = ollama_bin_dir / "ollama.exe"

    ollama_log = Path(str(exp_cfg["ollama_log_file"])).expanduser().resolve()
    ollama_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["OLLAMA_HOST"] = ollama_host

    info = {"ready": False, "started": False, "ollama_bin": str(ollama_bin), "ollama_log": str(ollama_log), "env": env}
    logger.info("Checking Ollama host=%s", ollama_host)
    if ollama_is_ready(ollama_host):
        info["ready"] = True
        logger.info("Ollama is already ready")
        return info

    if not bool(exp_cfg.get("auto_start_ollama", False)):
        logger.error("Ollama is not ready and auto_start_ollama=false")
        return info

    if not ollama_bin.exists():
        logger.error("Ollama binary not found: %s", ollama_bin)
        return info

    logger.info("Auto-starting Ollama: %s serve", ollama_bin)
    with ollama_log.open("a", encoding="utf-8") as logf:
        subprocess.Popen(
            [str(ollama_bin), "serve"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
    info["started"] = True

    for attempt in range(1, 16):
        time.sleep(1)
        if ollama_is_ready(ollama_host):
            info["ready"] = True
            logger.info("Ollama became ready after %d seconds", attempt)
            break
    if not info["ready"]:
        logger.error("Ollama did not become ready. See log: %s", ollama_log)
    return info


def load_dataset(dataset_path: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    logger.info("Loading dataset: %s", dataset_path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a JSON array: {dataset_path}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload:
        grouped[str(row["conversation_id"])].append(row)
    ordered: list[dict[str, Any]] = []
    for conv_id in sorted(grouped):
        ordered.extend(sorted(grouped[conv_id], key=lambda item: int(item.get("turn", 0))))

    dimension_counts = Counter(str(item.get("dimension", "unknown")) for item in ordered)
    logger.info("Dataset samples=%d conversations=%d", len(ordered), len(grouped))
    logger.info("Dimension distribution=%s", dict(sorted(dimension_counts.items())))
    return ordered


class LoggingQueryExpander:
    def __init__(self, inner: QueryExpander, logger: logging.Logger):
        self.inner = inner
        self.logger = logger
        self.last_variants: list[str] = []

    def expand(self, query: str, history: list[dict[str, str]] | None = None) -> tuple[list[str], bool]:
        variants, rewritten = self.inner.expand(query, history=history)
        self.last_variants = variants
        self.logger.info(
            "Query expansion input=%r history_messages=%d rewritten=%s variants=%s",
            query,
            len(history or []),
            rewritten,
            variants,
        )
        return variants, rewritten


def build_query_expander(config: dict[str, Any], logger: logging.Logger) -> LoggingQueryExpander:
    qe_cfg = config["query_expansion"]
    rewriter_type = str(qe_cfg.get("rewriter", "ollama"))
    logger.info("Initializing QueryExpander rewriter=%s model=%s", rewriter_type, qe_cfg.get("model_name"))
    if rewriter_type == "passthrough":
        rewriter = PassthroughRewriter()
    elif rewriter_type == "ollama":
        rewriter = OllamaRewriter(model_name=str(qe_cfg.get("model_name", "qwen2.5:32b")), temperature=0.0)
    else:
        rewriter = QwenRewriter()
    inner = QueryExpander(
        rewriter=rewriter,
        num_paraphrases=int(qe_cfg.get("num_paraphrases", 1)),
        add_translation=bool(qe_cfg.get("add_translation", True)),
        source_lang=str(qe_cfg.get("source_lang", "auto")),
        target_lang_for_translation=str(qe_cfg.get("target_lang_for_translation", "auto")),
    )
    return LoggingQueryExpander(inner, logger)


def capture_stdout_to_log(logger: logging.Logger, label: str):
    buffer = io.StringIO()

    @contextlib.contextmanager
    def _manager():
        with contextlib.redirect_stdout(buffer):
            yield
        text = buffer.getvalue().strip()
        if text:
            logger.info("%s stdout:\n%s", label, text)

    return _manager()


import re

def normalize_chunk_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()

    match = re.search(r'chunk_(\d+)', text)
    if match:
        return f"chunk_{match.group(1)}"

    if text.isdigit():
        return f"chunk_{text}"
    return text


def build_rag_components(config: dict[str, Any], exp_cfg: dict[str, Any], logger: logging.Logger):
    logger.info("Building index/retrieval state")
    state = ensure_index(
        doc_file=str(resolve_path(PROJECT_ROOT, exp_cfg["doc_file"])),
        url_file=None,
        persist_dir=str(resolve_path(PROJECT_ROOT, config["indexing"]["persist_dir"])),
        embedding_model="intfloat/multilingual-e5-base",
        force_rebuild=bool(config["indexing"].get("force_rebuild", False)),
        debug=bool(exp_cfg.get("debug_retrieval", False)),
    )
    logger.info("Index ready. Corpus chunks=%d", len(state["corpus"]))

    query_expander = build_query_expander(config, logger)
    reranker = CrossEncoderReranker(
        alpha=float(config["retrieval"].get("rerank_alpha", 0.5)),
        multi_variant_enabled=bool(config["retrieval"].get("multi_variant_rerank_enabled", False)),
    )
    retriever = HybridRetriever(
        vectordb=state["vectordb"],
        bm25_index=state["bm25_index"],
        tokenizer=state["tokenizer"],
        corpus=state["corpus"],
        reranker=reranker,
        weight_vec=float(config["retrieval"].get("weight_vec", 1.25)),
        weight_bm25=float(config["retrieval"].get("weight_bm25", 0.75)),
        original_weight=float(config["retrieval"].get("original_weight", 1.0)),
        paraphrase_weight=float(config["retrieval"].get("paraphrase_weight", 1.2)),
        translation_weight=float(config["retrieval"].get("translation_weight", 0.85)),
        rerank_candidates=int(exp_cfg.get("rerank_candidates", 10)),
        query_expander=query_expander,
        query_expansion_enabled=bool(config["query_expansion"].get("enabled", True)),
        multi_turn_enabled=bool(config["query_expansion"].get("multi_turn", {}).get("enabled", True)),
        max_history_turns=int(config["query_expansion"].get("multi_turn", {}).get("max_history_turns", 5)),
        debug=bool(exp_cfg.get("debug_retrieval", False)),
    )
    generator = AnswerGenerator(
        model_name=str(exp_cfg.get("generation_model", "qwen2.5:32b")),
        temperature=float(exp_cfg.get("generation_temperature", 0.0)),
        num_predict=int(exp_cfg.get("generation_num_predict", 256)),
    )
    logger.info("RAG components ready. rerank_candidates=%s generation_model=%s", exp_cfg.get("rerank_candidates"), exp_cfg.get("generation_model"))
    return retriever, generator, query_expander


def run_rag_with_rewrite(
    question: str,
    history: list[dict[str, str]],
    memory_context: str,
    retriever: HybridRetriever,
    generator: AnswerGenerator,
    query_expander: LoggingQueryExpander,
    logger: logging.Logger,
) -> tuple[str, list[str], list[str], list[str], str]:
    start = time.perf_counter()
    with capture_stdout_to_log(logger, "RAG"):
        docs = retriever.retrieve(question, history=history)
    contexts = [doc.page_content for doc in docs]
    chunk_ids = [normalize_chunk_id(doc.metadata.get("chunk_id")) for doc in docs]
    prompt = generator.render_prompt(question, docs, memory_context=memory_context)
    logger.info("Generation prompt for question=%r:\n%s", question, prompt)
    answer = str(generator.llm.invoke(prompt).content)
    elapsed = time.perf_counter() - start
    logger.info(
        "RAG completed question=%r elapsed=%.2fs variants=%s chunk_ids=%s answer=%r",
        question,
        elapsed,
        query_expander.last_variants,
        chunk_ids,
        answer,
    )
    return answer, contexts, chunk_ids, list(query_expander.last_variants), prompt


def keyword_hit_rate(answer: str, expected_keywords: list[Any]) -> float:
    keywords = [str(item).strip() for item in expected_keywords if str(item).strip()]
    if not keywords:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lower)
    return hits / len(keywords)


def retrieval_metrics(retrieved_ids: list[str], reference_ids: list[Any]) -> tuple[float, float, int]:
    retrieved = {normalize_chunk_id(item) for item in retrieved_ids if normalize_chunk_id(item)}
    reference = {normalize_chunk_id(item) for item in reference_ids if normalize_chunk_id(item)}
    hits = len(retrieved & reference)
    recall = hits / len(reference) if reference else 0.0
    precision = hits / len(retrieved) if retrieved else 0.0
    return recall, precision, hits


def init_ragas(exp_cfg: dict[str, Any], logger: logging.Logger):
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        logger.error("Missing dependency for Ragas evaluation: %s", exc)
        raise

    llm_model = str(exp_cfg.get("evaluation_llm_model", "qwen2.5:32b"))
    emb_model = str(exp_cfg.get("evaluation_embedding_model", "nomic-embed-text"))
    logger.info("Loading Ragas evaluation LLM: ChatOllama(model=%s, temperature=%s)", llm_model, exp_cfg.get("evaluation_llm_temperature"))
    eval_llm = ChatOllama(model=llm_model, temperature=float(exp_cfg.get("evaluation_llm_temperature", 0.0)), validate_model_on_init=True)
    logger.info("Loading Ragas embeddings: OllamaEmbeddings(model=%s)", emb_model)
    eval_embeddings = OllamaEmbeddings(model=emb_model)
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    logger.info("Ragas models loaded successfully")
    return ragas_llm, ragas_embeddings


def run_ragas_evaluation(rows: list[dict[str, Any]], exp_cfg: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_correctness, context_recall, faithfulness
    from ragas.run_config import RunConfig

    ragas_llm, ragas_embeddings = init_ragas(exp_cfg, logger)
    eval_items = [
        {
            "user_input": row["user_input"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rows
    ]
    logger.info("Creating Ragas EvaluationDataset rows=%d", len(eval_items))
    dataset = EvaluationDataset.from_list(eval_items)
    run_config = RunConfig(
        timeout=int(exp_cfg.get("ragas_timeout_seconds", 300)),
        max_workers=int(exp_cfg.get("ragas_max_workers", 1)),
    )
    logger.info(
        "Running Ragas evaluate metrics=context_recall,faithfulness,answer_correctness timeout=%ss max_workers=%s",
        run_config.timeout,
        run_config.max_workers,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, faithfulness, answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=bool(exp_cfg.get("ragas_raise_exceptions", False)),
    )
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if hasattr(result, "scores"):
        return pd.DataFrame(result.scores)
    return pd.DataFrame(result)


def combine_results(base_rows: list[dict[str, Any]], ragas_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    base_df = pd.DataFrame(base_rows)
    metric_cols = [col for col in ragas_df.columns if col not in {"user_input", "response", "retrieved_contexts", "reference"}]
    logger.info("Ragas result columns=%s", list(ragas_df.columns))
    for col in metric_cols:
        base_df[col] = pd.to_numeric(ragas_df[col], errors="coerce") if col in ragas_df else None
    return base_df


def save_results(df: pd.DataFrame, output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved detailed results: %s", output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    exp_cfg = deep_merge(DEFAULT_RAGAS_EXPERIMENT, config.get("ragas_experiment", {}))
    base_output_dir = resolve_path(PROJECT_ROOT, exp_cfg["output_dir"])
    output_dir = base_output_dir.parent / f"{base_output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(output_dir / str(exp_cfg["log_file"]))
    logger.info("Starting Ragas RAG evaluation")
    logger.info("Experiment config:\n%s", yaml.safe_dump(exp_cfg, allow_unicode=True, sort_keys=True))

    ollama_info = ensure_ollama(exp_cfg, logger)
    os.environ["OLLAMA_HOST"] = str(exp_cfg["ollama_host"])
    logger.info("Ollama info ready=%s started=%s bin=%s log=%s", ollama_info["ready"], ollama_info["started"], ollama_info["ollama_bin"], ollama_info["ollama_log"])

    dataset = load_dataset(resolve_path(PROJECT_ROOT, exp_cfg["dataset_path"]), logger)
    retriever, generator, query_expander = build_rag_components(config, exp_cfg, logger)

    base_rows: list[dict[str, Any]] = []
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dataset:
        conversations[str(item["conversation_id"])].append(item)

    for conv_id in sorted(conversations):
        history: list[dict[str, str]] = []
        memory = None
        if bool(config["memory"].get("enabled", True)):
            memory = ConversationMemory(
                rounds=int(config["memory"].get("short_term_rounds", 5)),
                model_name=str(exp_cfg.get("generation_model", config["generation"].get("model_name", "qwen2.5:32b"))),
                long_term_enabled=bool(config["memory"].get("long_term_enabled", False)),
            )
        turns = sorted(conversations[conv_id], key=lambda row: int(row.get("turn", 0)))
        logger.info(
            "Running conversation=%s turns=%d memory_enabled=%s short_term_rounds=%s",
            conv_id,
            len(turns),
            bool(memory),
            config["memory"].get("short_term_rounds", 5),
        )
        for item in turns:
            question = str(item["user_input"])
            turn = int(item.get("turn", 0))
            logger.info("Running sample conversation=%s turn=%d question=%r history_messages=%d", conv_id, turn, question, len(history))
            started = time.perf_counter()
            error = None
            prompt = ""
            variants: list[str] = []
            try:
                memory_context = memory.build_short_prompt() if memory else ""
                logger.info("Memory context for conversation=%s turn=%d:\n%s", conv_id, turn, memory_context or "<empty>")
                answer, contexts, chunk_ids, variants, prompt = run_rag_with_rewrite(
                    question, history, memory_context, retriever, generator, query_expander, logger
                )
            except Exception as exc:
                logger.exception("RAG failed conversation=%s turn=%d", conv_id, turn)
                error = f"RAG_ERROR: {type(exc).__name__}: {exc}"
                answer, contexts, chunk_ids = error, [], []
            elapsed = time.perf_counter() - started

            retrieval_recall, retrieval_precision, retrieval_hits = retrieval_metrics(chunk_ids, item.get("reference_chunk_ids", []))
            kw_rate = keyword_hit_rate(answer, item.get("expected_keywords", []))
            row = {
                "conversation_id": conv_id,
                "turn": turn,
                "dimension": item.get("dimension", "unknown"),
                "tags": item.get("tags", []),
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "retrieved_chunk_ids": chunk_ids,
                "reference": item.get("reference", ""),
                "reference_chunk_ids": item.get("reference_chunk_ids", []),
                "expected_keywords": item.get("expected_keywords", []),
                "expanded_queries": variants,
                "generation_prompt": prompt,
                "rag_elapsed_seconds": elapsed,
                "retrieval_recall": retrieval_recall,
                "retrieval_precision": retrieval_precision,
                "retrieval_hit_count": retrieval_hits,
                "keyword_hit_rate": kw_rate,
                "error": error,
            }
            base_rows.append(row)
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            if memory:
                memory.add_turn(question, answer)
            logger.info(
                "Sample done conversation=%s turn=%d elapsed=%.2fs retrieval_recall=%.3f retrieval_precision=%.3f keyword_hit_rate=%.3f",
                conv_id,
                turn,
                elapsed,
                retrieval_recall,
                retrieval_precision,
                kw_rate,
            )

    try:
        ragas_df = run_ragas_evaluation(base_rows, exp_cfg, logger)
        final_df = combine_results(base_rows, ragas_df, logger)
    except Exception as exc:
        logger.exception("Ragas evaluation failed; saving custom metrics only")
        final_df = pd.DataFrame(base_rows)
        final_df["ragas_error"] = f"{type(exc).__name__}: {exc}"

    numeric_cols = final_df.select_dtypes(include="number").columns.tolist()
    summary = final_df.groupby("dimension")[numeric_cols].mean(numeric_only=True).reset_index()
    logger.info("Summary by dimension:\n%s", summary.to_string(index=False))
    print("\nSummary by dimension:")
    print(summary.to_string(index=False))

    save_results(final_df, output_dir / str(exp_cfg["results_file"]), logger)
    logger.info("Experiment finished")


if __name__ == "__main__":
    main()
