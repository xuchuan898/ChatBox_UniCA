"""Run short-term memory regression tests for ChatBox_UniCA.

The script builds the same core RAG components used by chat_box.py, runs each
multi-turn conversation with memory kept across turns, optionally runs a
reset-each-turn baseline, evaluates configured checks with deterministic
keyword rules, and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUN_LOG_HANDLE: Any | None = None
RUN_DIR: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)
    if RUN_LOG_HANDLE:
        RUN_LOG_HANDLE.write(message + "\n")
        RUN_LOG_HANDLE.flush()


def log_block(title: str, content: str) -> None:
    log(f"{title}_BEGIN")
    for line in content.splitlines():
        log(line)
    log(f"{title}_END")


@dataclass
class Pipeline:
    retriever: Any
    generator: Any
    cache: Any | None
    config: dict[str, Any]

    def new_memory(self, memory_file: Path) -> Any:
        from core.conversation_memory import ConversationMemory

        memory = ConversationMemory(
            rounds=int(self.config["memory"]["short_term_rounds"]),
            memory_file=memory_file,
            model_name=str(self.config["generation"]["model_name"]),
        )
        # The experiment is scoped to short-term memory. Disabling extraction
        # avoids persistent long-term facts changing boundary-overflow results.
        memory._extract_memory = lambda user, assistant: None  # type: ignore[method-assign]
        return memory

    @staticmethod
    def serialize_doc(doc: Any, rank: int) -> dict[str, Any]:
        content = str(getattr(doc, "page_content", ""))
        metadata = dict(getattr(doc, "metadata", {}) or {})
        return {
            "rank": rank,
            "chunk_id": metadata.get("chunk_id"),
            "source": metadata.get("source"),
            "chunk_type": metadata.get("chunk_type"),
            "preview": " ".join(content.split())[:500],
            "content": content,
        }

    @staticmethod
    def memory_history(memory: Any | None) -> list[dict[str, str]]:
        if not memory:
            return []
        history = []
        for turn in getattr(memory, "short_term", []):
            user = str(turn.get("user", "")).strip()
            assistant = str(turn.get("assistant", "")).strip()
            if user:
                history.append({"role": "user", "content": user})
            if assistant:
                history.append({"role": "assistant", "content": assistant})
        return history

    def run_one(self, question: str, memory: Any | None) -> dict[str, Any]:
        memory_context = ""
        if memory:
            memory_context = memory.build_short_prompt()

        if self.cache:
            hit = self.cache.get(question)
            if hit:
                answer = str(hit.get("answer", ""))
                if memory:
                    memory.add_turn(question, answer)
                return {
                    "answer": answer,
                    "memory_context": memory_context,
                    "retrieved_docs": [],
                    "prompt": "[CACHE HIT] Prompt was not rendered.",
                    "cache_hit": True,
                }

        docs = self.retriever.retrieve(question, history=self.memory_history(memory))
        prompt = self.generator.render_prompt(question, docs, memory_context)
        answer = str(self.generator.llm.invoke(prompt).content)
        if self.cache:
            self.cache.put(question, answer, [{"chunk_id": d.metadata.get("chunk_id")} for d in docs])
        if memory:
            memory.add_turn(question, answer)
        return {
            "answer": answer,
            "memory_context": memory_context,
            "retrieved_docs": [self.serialize_doc(doc, idx) for idx, doc in enumerate(docs, 1)],
            "prompt": prompt,
            "cache_hit": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short-term memory tests.")
    parser.add_argument("--dataset", type=Path, default=Path("questions/test_memory_short_term.json"))
    parser.add_argument("--doc-file", type=Path, default=Path("docs/chroma/master.md"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset/report generation without LLM calls.")
    parser.add_argument("--skip-reset-baseline", action="store_true", help="Do not run the reset-each-turn comparison.")
    parser.add_argument("--disable-cache", action="store_true", default=True, help="Disable semantic cache for test isolation.")
    parser.add_argument("--enable-query-expansion", action="store_true", help="Use configured query expansion.")
    parser.add_argument(
        "--ollama-bin-dir",
        type=Path,
        default=Path("~/ollama/bin").expanduser(),
        help="Directory containing the ollama binary.",
    )
    parser.add_argument(
        "--ollama-log-file",
        type=Path,
        default=Path("~/ollama/ollama.log").expanduser(),
        help="Log file used when auto-starting ollama serve.",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama host URL used for readiness checks.",
    )
    parser.add_argument(
        "--auto-start-ollama",
        action="store_true",
        default=True,
        help="Auto-start ollama serve in background if the host is unreachable.",
    )
    parser.add_argument(
        "--no-auto-start-ollama",
        action="store_false",
        dest="auto_start_ollama",
        help="Do not auto-start ollama serve if the host is unreachable.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Model to check/pull before running. Defaults to generation.model_name from config.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(output_dir) / f"memory_test_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_inputs(run_dir: Path, dataset_path: Path, doc_path: Path, config_path: Path) -> None:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (dataset_path, "test_memory_short_term.json"),
        (doc_path, "master.md"),
        (config_path, "config.yaml"),
    ]:
        if src.exists():
            shutil.copy2(src, inputs_dir / name)


def ollama_is_ready(ollama_host: str) -> bool:
    try:
        with urlopen(f"{ollama_host.rstrip('/')}/api/tags", timeout=2) as _:
            return True
    except (URLError, TimeoutError, OSError):
        return False


def resolve_ollama_binary(ollama_bin_dir: Path) -> Path:
    plain = ollama_bin_dir / "ollama"
    if plain.exists():
        return plain
    found = shutil.which("ollama")
    if found:
        return Path(found)
    return plain


def ensure_ollama(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    ollama_bin_dir = args.ollama_bin_dir.expanduser().resolve()
    ollama_bin = resolve_ollama_binary(ollama_bin_dir)
    ollama_log = args.ollama_log_file.expanduser().resolve()
    ollama_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_NUM_GPU"] = "1"

    info = {
        "env": env,
        "ollama_bin": str(ollama_bin),
        "ollama_log": str(ollama_log),
        "started": False,
        "ready": False,
        "model_available": False,
        "model_pulled": False,
    }

    log(f"[INFO] PATH prepended with: {ollama_bin_dir}")
    log(f"[INFO] Ollama host: {args.ollama_host}")
    log(f"[INFO] Ollama binary: {ollama_bin}")
    log(f"[INFO] Ollama log: {ollama_log}")
    log(f"[INFO] Checking Ollama readiness at {args.ollama_host}/api/tags")

    if ollama_is_ready(args.ollama_host):
        info["ready"] = True
        log("[INFO] Ollama is already ready")
    elif args.auto_start_ollama:
        if not ollama_bin.exists():
            log(f"[WARN] Ollama binary not found: {ollama_bin}")
            return info
        log("[INFO] Ollama is not ready; starting `ollama serve` in background")
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
        for second in range(1, 31):
            if ollama_is_ready(args.ollama_host):
                info["ready"] = True
                log(f"[INFO] Ollama became ready after {second}s")
                break
            if second == 1 or second % 5 == 0:
                log(f"[INFO] Waiting for Ollama... {second}s")
            time.sleep(1)
    else:
        log("[WARN] Ollama is not ready. Pass --auto-start-ollama to start it from this script.")
        return info

    if not info["ready"]:
        log("[WARN] Ollama did not become ready; live model calls will fail.")
        return info

    try:
        log(f"[INFO] Checking Ollama model availability: {model_name}")
        result = subprocess.run([str(ollama_bin), "list"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info["model_available"] = model_name in result.stdout
        if info["model_available"]:
            log(f"[INFO] Model is available: {model_name}")
        else:
            log(f"[INFO] Model not found locally; pulling: {model_name}")
            pull_result = subprocess.run([str(ollama_bin), "pull", model_name], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            info["model_pulled"] = pull_result.returncode == 0
            info["model_available"] = info["model_pulled"]
            log(f"[INFO] Model pull finished: success={info['model_pulled']}")
            if pull_result.returncode != 0:
                log(f"[WARN] ollama pull stderr: {pull_result.stderr.strip()}")
    except Exception as exc:
        log(f"[WARN] Could not verify/pull Ollama model: {exc}")

    os.environ.update(env)
    return info


def configured_model_name(config_path: Path, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    try:
        from core.config_loader import load_config

        config = load_config(config_path)
        return str(config["generation"]["model_name"])
    except Exception:
        return "gemma3:4b"


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    from chat_box import _build_query_expander, prepare_data
    from core.config_loader import apply_cli_overrides, load_config
    from core.generator import AnswerGenerator
    from core.reranker import CrossEncoderReranker
    from core.retriever import HybridRetriever
    from core.semantic_cache import SemanticCache

    log("[INFO] Loading config")
    config = load_config(resolve(args.config))
    config = apply_cli_overrides(
        config,
        {
            "cache.enabled": False if args.disable_cache else None,
            "query_expansion.enabled": True if args.enable_query_expansion else False,
            "memory.enabled": True,
            "memory.short_term_rounds": 5,
        },
    )
    log(
        "[INFO] Effective settings: "
        f"model={config['generation']['model_name']} "
        f"cache={config['cache']['enabled']} "
        f"query_expansion={config['query_expansion']['enabled']} "
        f"memory_rounds={config['memory']['short_term_rounds']}"
    )
    log(f"[INFO] Preparing index/data from {resolve(args.doc_file)}")
    start = time.perf_counter()
    vectordb, bm25_index, tokenizer, corpus = prepare_data(str(resolve(args.doc_file)), debug=args.debug)
    log(f"[INFO] Data prepared in {time.perf_counter() - start:.2f}s; corpus chunks={len(corpus)}")
    log("[INFO] Building query expander, reranker, retriever, and generator")
    query_expander = _build_query_expander(config)
    reranker = CrossEncoderReranker(
        alpha=float(config["retrieval"]["rerank_alpha"]),
        multi_variant_enabled=bool(config["retrieval"].get("multi_variant_rerank_enabled", False)),
    )
    retriever = HybridRetriever(
        vectordb=vectordb,
        bm25_index=bm25_index,
        tokenizer=tokenizer,
        corpus=corpus,
        reranker=reranker,
        weight_vec=float(config["retrieval"]["weight_vec"]),
        weight_bm25=float(config["retrieval"]["weight_bm25"]),
        rerank_candidates=int(config["retrieval"]["rerank_candidates"]),
        query_expander=query_expander,
        query_expansion_enabled=bool(config["query_expansion"]["enabled"]),
        multi_turn_enabled=bool(config["query_expansion"].get("multi_turn", {}).get("enabled", True)),
        max_history_turns=int(config["query_expansion"].get("multi_turn", {}).get("max_history_turns", 5)),
        debug=args.debug,
    )
    generator = AnswerGenerator(
        model_name=str(config["generation"]["model_name"]),
        temperature=float(config["generation"]["temperature"]),
        num_predict=int(config["generation"]["num_predict"]),
    )
    cache = None
    if config["cache"]["enabled"]:
        cache = SemanticCache(threshold=float(config["cache"]["threshold"]), ttl=int(config["cache"]["ttl"]), debug=args.debug)
    log("[INFO] Pipeline ready")
    return Pipeline(retriever=retriever, generator=generator, cache=cache, config=config)


def run_conversation(
    pipeline: Pipeline | None,
    conversation: dict[str, Any],
    mode: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    memory_base = RUN_DIR if RUN_DIR else PROJECT_ROOT / "experiments" / "results"
    memory_dir = memory_base / "memory_state"
    memory_dir.mkdir(parents=True, exist_ok=True)
    shared_memory = None
    if not dry_run and pipeline and mode == "keep_memory":
        shared_memory = pipeline.new_memory(memory_dir / f"{conversation['id']}_{mode}.json")

    total_turns = len(conversation["turns"])
    log(f"[CONV] {conversation['id']} mode={mode} turns={total_turns}")
    for idx, turn in enumerate(conversation["turns"], 1):
        question = str(turn["text"])
        log(f"[TURN] {conversation['id']} {mode} {idx}/{total_turns}: {question}")
        start = time.perf_counter()
        if dry_run:
            trace = {
                "answer": "[DRY RUN] No model call was executed.",
                "memory_context": "",
                "retrieved_docs": [],
                "prompt": "[DRY RUN] Prompt was not rendered.",
                "cache_hit": False,
            }
        else:
            if pipeline is None:
                raise RuntimeError("Pipeline is required unless --dry-run is set.")
            if mode == "reset_each_turn":
                memory = pipeline.new_memory(memory_dir / f"{conversation['id']}_{mode}_{idx}.json")
            else:
                memory = shared_memory
            trace = pipeline.run_one(question, memory)
        elapsed = time.perf_counter() - start
        log(f"[TURN] {conversation['id']} {mode} {idx}/{total_turns} finished in {elapsed:.2f}s")
        log(f"[ANSWER] {conversation['id']} {mode} {idx}/{total_turns}: {trace['answer']}")
        log_block(f"[MEMORY_CONTEXT] {conversation['id']} {mode} {idx}/{total_turns}", trace["memory_context"] or "[EMPTY]")
        retrieval_lines = []
        for doc in trace["retrieved_docs"]:
            retrieval_lines.append(
                f"Rank {doc['rank']} | chunk_id={doc.get('chunk_id')} | source={doc.get('source')} | "
                f"type={doc.get('chunk_type')}\n{doc.get('preview')}"
            )
        log_block(f"[RETRIEVAL] {conversation['id']} {mode} {idx}/{total_turns}", "\n\n".join(retrieval_lines) or "[NO DOCS]")
        log_block(f"[PROMPT] {conversation['id']} {mode} {idx}/{total_turns}", trace["prompt"])
        logs.append(
            {
                "turn": idx,
                "question": question,
                "answer": trace["answer"],
                "elapsed_sec": elapsed,
                "memory_context": trace["memory_context"],
                "retrieved_docs": trace["retrieved_docs"],
                "prompt": trace["prompt"],
                "cache_hit": trace["cache_hit"],
            }
        )
    return logs


def contains_any(answer: str, terms: list[str]) -> bool:
    lower_answer = answer.lower()
    return any(term.lower() in lower_answer for term in terms)


def evaluate_conversation(conversation: dict[str, Any], logs: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    checks = []
    passed = 0
    for check in conversation.get("checks", []):
        turn_no = int(check["turn"])
        answer = next((item["answer"] for item in logs if int(item["turn"]) == turn_no), "")
        if dry_run:
            ok = False
            reason = "Skipped in dry-run mode."
        else:
            expected = bool(check["expected"])
            terms = [str(term) for term in check.get("must_include_any", [])]
            matched = contains_any(answer, terms) if terms else bool(answer.strip())
            ok = matched if expected else not matched
            relation = "include any of" if expected else "not include any of"
            reason = f"Expected answer to {relation}: {', '.join(terms)}"
        if ok:
            passed += 1
        checks.append({"name": check["name"], "passed": ok, "reason": reason})
    total = len(checks)
    return {
        "conversation_passed": total > 0 and passed == total,
        "passed_checks": passed,
        "total_checks": total,
        "checks": checks,
    }


def collect_results(args: argparse.Namespace, dataset: dict[str, Any]) -> dict[str, Any]:
    pipeline = None if args.dry_run else build_pipeline(args)
    dimensions = []
    total_conversations = 0
    passed_conversations = 0

    total_dimensions = len(dataset["dimensions"])
    for dim_idx, dimension in enumerate(dataset["dimensions"], 1):
        log(f"[DIM] {dim_idx}/{total_dimensions} start: {dimension['dimension']}")
        dim_results = []
        dim_passed = 0
        total_dim_convs = len(dimension["conversations"])
        for conv_idx, conversation in enumerate(dimension["conversations"], 1):
            log(f"[CONV] {dimension['dimension']} {conv_idx}/{total_dim_convs} start: {conversation['id']}")
            keep_logs = run_conversation(pipeline, conversation, "keep_memory", args.dry_run)
            evaluation = evaluate_conversation(conversation, keep_logs, args.dry_run)
            log(
                f"[EVAL] {conversation['id']} keep_memory "
                f"checks={evaluation['passed_checks']}/{evaluation['total_checks']} "
                f"passed={evaluation['conversation_passed']}"
            )
            reset_logs = []
            if not args.skip_reset_baseline:
                reset_logs = run_conversation(pipeline, conversation, "reset_each_turn", args.dry_run)
            if evaluation["conversation_passed"]:
                dim_passed += 1
                passed_conversations += 1
            total_conversations += 1
            dim_results.append(
                {
                    "id": conversation["id"],
                    "tags": conversation.get("tags", []),
                    "keep_memory_logs": keep_logs,
                    "reset_each_turn_logs": reset_logs,
                    "evaluation": evaluation,
                }
            )
            log(f"[CONV] {conversation['id']} done")
        dimensions.append(
            {
                "dimension": dimension["dimension"],
                "description": dimension["description"],
                "passed": dim_passed,
                "total": len(dimension["conversations"]),
                "conversations": dim_results,
            }
        )
        log(f"[DIM] {dimension['dimension']} done: {dim_passed}/{len(dimension['conversations'])} conversations passed")
    return {
        "meta": dataset["meta"],
        "dry_run": args.dry_run,
        "total_conversations": total_conversations,
        "passed_conversations": passed_conversations,
        "dimensions": dimensions,
    }


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0%"
    return f"{round((numerator / denominator) * 100)}%"


def truncate(text: str, limit: int = 1800) -> str:
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 20].rstrip() + "\n...[truncated]"


def render_docs_for_report(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "[NO DOCS]"
    lines = []
    for doc in docs:
        lines.append(
            f"Rank {doc['rank']} | chunk_id={doc.get('chunk_id')} | source={doc.get('source')} | type={doc.get('chunk_type')}\n"
            f"{truncate(str(doc.get('content', '')), 1200)}"
        )
    return "\n\n".join(lines)


def render_report(results: dict[str, Any], full_log_file: Path | None = None) -> str:
    total = int(results["total_conversations"])
    passed = int(results["passed_conversations"])
    date = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# Short-term Memory Test Report",
        "",
        "## Summary",
        f"- Total conversations: {total}",
        f"- Passed: {passed} / {total} ({pct(passed, total)})",
        f"- Date: {date}",
        f"- Mode: {'dry-run' if results['dry_run'] else 'live RAG calls'}",
        f"- Full log: {full_log_file if full_log_file else '[not configured]'}",
        "",
        "## By Dimension",
        "",
        "| Dimension | Pass | Total | Rate |",
        "|-----------|------|-------|------|",
    ]
    for dim in results["dimensions"]:
        lines.append(f"| {dim['dimension']} | {dim['passed']} | {dim['total']} | {pct(dim['passed'], dim['total'])} |")

    lines.extend(["", "## Overall Pass / Fail", "", "| Conversation | Result | Checks |", "|--------------|--------|--------|"])
    for dim in results["dimensions"]:
        for conv in dim["conversations"]:
            ev = conv["evaluation"]
            result = "PASS" if ev["conversation_passed"] else "FAIL"
            lines.append(f"| {conv['id']} | {result} | {ev['passed_checks']} / {ev['total_checks']} |")

    lines.extend(["", "## Detailed Logs"])
    for dim in results["dimensions"]:
        lines.extend(["", f"### {dim['dimension']}"])
        for conv in dim["conversations"]:
            ev = conv["evaluation"]
            result = "PASS" if ev["conversation_passed"] else "FAIL"
            lines.extend(["", f"#### {conv['id']} - {result}", "", "**Checks**"])
            for check in ev["checks"]:
                marker = "PASS" if check["passed"] else "FAIL"
                lines.append(f"- {marker}: {check['name']} ({check['reason']})")
            lines.extend(["", "**Keep-memory run**"])
            for item in conv["keep_memory_logs"]:
                lines.append(f"- Turn {item['turn']} question: {item['question']}")
                lines.append(f"- Turn {item['turn']} answer ({item['elapsed_sec']:.2f}s): {truncate(item['answer'])}")
                lines.extend(["", "<details><summary>Retrieval, Prompt, and Memory</summary>", ""])
                lines.extend(["Memory context:", "```text", truncate(item.get("memory_context", ""), 3000) or "[EMPTY]", "```"])
                lines.extend(["Retrieval:", "```text", render_docs_for_report(item.get("retrieved_docs", [])), "```"])
                lines.extend(["Prompt:", "```text", truncate(item.get("prompt", ""), 6000), "```", "</details>", ""])
            if conv["reset_each_turn_logs"]:
                lines.extend(["", "**Reset-each-turn baseline**"])
                for item in conv["reset_each_turn_logs"]:
                    lines.append(f"- Turn {item['turn']} answer ({item['elapsed_sec']:.2f}s): {truncate(item['answer'], 800)}")
                    lines.extend(["", "<details><summary>Reset baseline trace</summary>", ""])
                    lines.extend(["Retrieval:", "```text", render_docs_for_report(item.get("retrieved_docs", [])), "```"])
                    lines.extend(["Prompt:", "```text", truncate(item.get("prompt", ""), 4000), "```", "</details>", ""])

    lines.extend(
        [
            "",
            "## Findings & Recommendations",
            "- Treat reset-each-turn results as a baseline: memory-sensitive turns should usually degrade there while direct factual retrieval may still pass.",
            "- Boundary-overflow failures can be legitimate if retrieval answers from the source document even after the short-term deque has forgotten the first turn.",
            "- For stricter memory-only overflow tests, add user-introduced aliases or facts that are not present in the source document.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    global RUN_DIR, RUN_LOG_HANDLE
    args = parse_args()
    start = time.perf_counter()
    RUN_DIR = make_run_dir(args.output_dir)
    dataset_path = resolve(args.dataset)
    doc_path = resolve(args.doc_file)
    config_path = resolve(args.config)
    report_file = resolve(args.report_file) if args.report_file else RUN_DIR / "memory_test_report.md"
    full_log_file = resolve(args.log_file) if args.log_file else RUN_DIR / "memory_test_full.log"
    snapshot_inputs(RUN_DIR, dataset_path, doc_path, config_path)
    full_log_file.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_HANDLE = full_log_file.open("w", encoding="utf-8")
    log(f"[INFO] Project root: {PROJECT_ROOT}")
    log(f"[INFO] Run dir: {RUN_DIR}")
    log(f"[INFO] Input snapshot dir: {RUN_DIR / 'inputs'}")
    log(f"[INFO] Dataset: {dataset_path}")
    log(f"[INFO] Doc file: {doc_path}")
    log(f"[INFO] Report file: {report_file}")
    log(f"[INFO] Full log file: {full_log_file}")
    log(f"[INFO] Dry run: {args.dry_run}")
    dataset = load_dataset(dataset_path)
    total_conversations = sum(len(dim["conversations"]) for dim in dataset["dimensions"])
    log(f"[INFO] Loaded dataset: dimensions={len(dataset['dimensions'])}, conversations={total_conversations}")
    if not args.dry_run:
        model_name = configured_model_name(config_path, args.ollama_model)
        ollama_info = ensure_ollama(args, model_name)
        log(
            "[INFO] Ollama status: "
            f"ready={ollama_info['ready']} started={ollama_info['started']} "
            f"model_available={ollama_info['model_available']}"
        )
    results = collect_results(args, dataset)
    log("[INFO] Rendering report")
    report = render_report(results, full_log_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    log(f"[INFO] Memory test report written to {report_file}")
    log_block("[REPORT]", report)
    log(f"[INFO] Finished in {time.perf_counter() - start:.2f}s")
    if RUN_LOG_HANDLE:
        RUN_LOG_HANDLE.close()
        RUN_LOG_HANDLE = None


if __name__ == "__main__":
    main()
