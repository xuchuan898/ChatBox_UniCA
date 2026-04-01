import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_box import chatbox, prepare_data

PROMPT_TEMPLATE_TEXT = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use the context to answer concisely. Keep the answer as concise as possible. "
    "Always say \"Merci pour votre question!\" at the end of the answer. "
    "Please answer the question in the langugae used by the question\n"
    "{context}\n\n"
    "Question: {input}\n"
    "Answer:"
)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
SKIP_FILES = {"source_urls.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full question x document matrix experiment and export analyzable artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path (default: auto-detected).",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs/chroma"),
        help="Directory containing source docs to evaluate.",
    )
    parser.add_argument(
        "--question-catalog",
        type=Path,
        default=Path("questions/generated_questions_docs_chroma.json"),
        help="JSON file with 5 questions per document.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Output base directory for experiment artifacts.",
    )
    parser.add_argument(
        "--ollama-bin-dir",
        type=Path,
        default=Path("~/ollama/bin").expanduser(),
        help="Directory containing ollama binary.",
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
        help="Auto-start ollama serve in background if host is unreachable.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output from chat pipeline.")
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def ollama_is_ready(ollama_host: str) -> bool:
    try:
        with urlopen(f"{ollama_host.rstrip('/')}/api/tags", timeout=2) as _:
            return True
    except (URLError, TimeoutError, OSError):
        return False


def ensure_ollama(args: argparse.Namespace) -> dict:
    ollama_bin_dir = args.ollama_bin_dir.expanduser().resolve()
    ollama_bin = ollama_bin_dir / "ollama"
    if os.name == "nt" and not ollama_bin.exists():
        ollama_bin = ollama_bin_dir / "ollama.exe"

    ollama_log = args.ollama_log_file.expanduser().resolve()
    ollama_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["OLLAMA_HOST"] = args.ollama_host

    info = {
        "env": env,
        "ollama_bin": str(ollama_bin),
        "ollama_log": str(ollama_log),
        "started": False,
        "ready": False,
    }

    if ollama_is_ready(args.ollama_host):
        info["ready"] = True
        return info

    if not args.auto_start_ollama:
        return info

    if not ollama_bin.exists():
        return info

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

    for _ in range(15):
        if ollama_is_ready(args.ollama_host):
            info["ready"] = True
            break
        time.sleep(1)

    return info


def slugify_doc(path: Path) -> str:
    return f"{path.stem}_{path.suffix.lstrip('.').lower()}"


def discover_docs(docs_dir: Path) -> list[Path]:
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(path)
    return docs


def load_question_catalog(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "documents" not in payload:
        raise ValueError(f"Invalid question catalog format: missing 'documents' key in {path}")
    return payload


def flatten_questions(question_catalog: dict) -> list[dict]:
    flat = []
    q_global = 1
    for doc_name, questions in question_catalog["documents"].items():
        for idx, question in enumerate(questions, start=1):
            flat.append(
                {
                    "q_global_id": q_global,
                    "source_doc": doc_name,
                    "source_q_id": idx,
                    "question": question,
                }
            )
            q_global += 1
    return flat


def build_doc_question_plan(docs: list[Path], question_catalog: dict) -> tuple[dict[str, list[dict]], int]:
    plan: dict[str, list[dict]] = {}
    q_global = 1
    total = 0
    for doc in docs:
        doc_name = doc.name
        doc_questions = question_catalog["documents"].get(doc_name, [])
        rows = []
        for idx, question in enumerate(doc_questions, start=1):
            rows.append(
                {
                    "q_global_id": q_global,
                    "source_doc": doc_name,
                    "source_q_id": idx,
                    "question": question,
                }
            )
            q_global += 1
        plan[doc_name] = rows
        total += len(rows)
    return plan, total


def serialize_retrieval_docs(retrieved_docs: list, max_preview_chars: int = 220) -> list[dict]:
    rows = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        rows.append(
            {
                "rank": idx,
                "source": doc.metadata.get("source", "unknown"),
                "source_type": doc.metadata.get("source_type", "unknown"),
                "preview": doc.page_content[:max_preview_chars].replace("\n", " "),
            }
        )
    return rows


def render_prompt(context_docs: list, question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in context_docs)
    return PROMPT_TEMPLATE_TEXT.format(context=context, input=question)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_comparison_csv(path: Path, responses: list[dict], doc_tags: list[str]) -> None:
    by_question = {}
    for row in responses:
        q_id = row["q_global_id"]
        if q_id not in by_question:
            by_question[q_id] = {
                "q_global_id": q_id,
                "source_doc": row["source_doc"],
                "source_q_id": row["source_q_id"],
                "question": row["question"],
            }
        by_question[q_id][f"answer__{row['target_doc_tag']}"] = row["answer"]

    fieldnames = ["q_global_id", "source_doc", "source_q_id", "question"] + [
        f"answer__{tag}" for tag in doc_tags
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for q_id in sorted(by_question):
            out = by_question[q_id]
            for tag in doc_tags:
                out.setdefault(f"answer__{tag}", "")
            writer.writerow(out)


def build_summary(responses: list[dict], docs: list[Path], run_dir: Path) -> dict:
    total_runs = len(responses)
    qa_times = [r["qa_time_sec"] for r in responses]
    retrieval_times = [r["retrieval_time_sec"] for r in responses]

    per_target = {}
    for doc in docs:
        tag = slugify_doc(doc)
        rows = [r for r in responses if r["target_doc_tag"] == tag]
        if not rows:
            continue
        per_target[tag] = {
            "doc_path": str(doc),
            "runs": len(rows),
            "avg_answer_chars": round(statistics.mean(len(r["answer"]) for r in rows), 2),
            "avg_qa_time_sec": round(statistics.mean(r["qa_time_sec"] for r in rows), 3),
            "avg_retrieval_time_sec": round(statistics.mean(r["retrieval_time_sec"] for r in rows), 3),
            "merci_suffix_ratio": round(
                sum("Merci pour votre question!" in r["answer"] for r in rows) / len(rows), 3
            ),
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "run_mode": "doc_aligned",
        "total_runs": total_runs,
        "total_docs": len(docs),
        "total_questions": len({r['q_global_id'] for r in responses}),
        "avg_qa_time_sec": round(statistics.mean(qa_times), 3) if qa_times else 0.0,
        "avg_retrieval_time_sec": round(statistics.mean(retrieval_times), 3) if retrieval_times else 0.0,
        "per_target_doc": per_target,
    }


def write_summary_md(path: Path, summary: dict, docs: list[Path], question_catalog_path: Path) -> None:
    lines = [
        "# Document Matrix Experiment Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Question catalog: {question_catalog_path}",
        f"- Total docs: {summary['total_docs']}",
        f"- Total questions: {summary['total_questions']}",
        f"- Total runs: {summary['total_runs']}",
        f"- Avg retrieval time (s): {summary['avg_retrieval_time_sec']}",
        f"- Avg QA time (s): {summary['avg_qa_time_sec']}",
        "",
        "## Docs",
        "",
    ]
    for doc in docs:
        lines.append(f"- {doc}")

    lines.extend(["", "## Per-Document Metrics", "", "| Doc Tag | Runs | Avg Answer Chars | Avg QA(s) | Merci Ratio |", "| --- | ---: | ---: | ---: | ---: |"])  # noqa: E501

    for tag, metrics in summary["per_target_doc"].items():
        lines.append(
            f"| {tag} | {metrics['runs']} | {metrics['avg_answer_chars']} | "
            f"{metrics['avg_qa_time_sec']} | {metrics['merci_suffix_ratio']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `responses.jsonl`: each QA call with full rendered prompt, retrieval traces, and answer.",
            "- `prompts.jsonl`: prompt-only view for quick external AI analysis.",
            "- `comparison.csv`: side-by-side answers for each question across docs.",
            "- `summary.json`: machine-readable aggregate metrics.",
            "- `analysis_prompt.txt`: ready-to-use instruction text for asking another AI to analyze this run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_prompt_txt(path: Path, summary: dict, docs: list[Path], question_catalog_path: Path) -> None:
    doc_list = "\n".join(f"- {doc.name}" for doc in docs)
    prompt_text = (
        "You are evaluating a RAG cross-document QA experiment.\n"
        "Please analyze the experiment outputs and provide a structured report.\n\n"
        "Files to use from this run directory:\n"
        "- summary.json\n"
        "- comparison.csv\n"
        "- responses.jsonl\n"
        "- prompts.jsonl\n"
        "- question_catalog_snapshot.json\n\n"
        "Run context:\n"
        f"- Question catalog: {question_catalog_path.name}\n"
        f"- Total docs: {summary['total_docs']}\n"
        f"- Total questions: {summary['total_questions']}\n"
        f"- Total runs: {summary['total_runs']}\n"
        f"- Avg retrieval time (s): {summary['avg_retrieval_time_sec']}\n"
        f"- Avg QA time (s): {summary['avg_qa_time_sec']}\n"
        "- Target docs:\n"
        f"{doc_list}\n\n"
        "Deliverables (required):\n"
        "1) Executive summary (5-8 bullets).\n"
        "2) Document-level findings: strengths, weaknesses, likely causes.\n"
        "3) Cross-document comparison on answer quality, consistency, and relevance.\n"
        "4) Prompt and retrieval analysis: identify failure patterns from prompts/retrieval traces.\n"
        "5) Quantitative observations from summary.json and comparison.csv.\n"
        "6) Top 10 most problematic QA cases with q_global_id, target_doc_tag, and explanation.\n"
        "7) Action plan with prioritized improvements (quick wins vs deeper fixes).\n"
        "8) Suggested next experiment design and additional metrics to collect.\n\n"
        "Output format:\n"
        "- Use Markdown.\n"
        "- Include short tables where useful.\n"
        "- Reference rows by q_global_id and target_doc_tag when citing examples.\n"
    )
    path.write_text(prompt_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    docs_dir = resolve_path(project_root, args.docs_dir).resolve()
    question_catalog_path = resolve_path(project_root, args.question_catalog).resolve()
    output_base = resolve_path(project_root, args.output_dir).resolve()
    ollama_info = ensure_ollama(args)
    os.environ["OLLAMA_HOST"] = args.ollama_host

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs dir not found: {docs_dir}")
    if not question_catalog_path.exists():
        raise FileNotFoundError(f"Question catalog not found: {question_catalog_path}")

    docs = discover_docs(docs_dir)
    if not docs:
        raise ValueError(f"No supported docs found in: {docs_dir}")

    question_catalog = load_question_catalog(question_catalog_path)
    questions = flatten_questions(question_catalog)
    if not questions:
        raise ValueError(f"No questions found in catalog: {question_catalog_path}")
    doc_question_plan, planned_runs = build_doc_question_plan(docs, question_catalog)
    if planned_runs == 0:
        raise ValueError("No document-aligned questions found for discovered docs.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"doc_matrix_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    responses = []
    prompt_rows = []

    print(f"[INFO] Project root: {project_root}", flush=True)
    print(f"[INFO] Docs: {len(docs)}", flush=True)
    print(f"[INFO] Questions in catalog: {len(questions)}", flush=True)
    print(f"[INFO] Run mode: doc_aligned (each doc uses only its own questions)", flush=True)
    print(f"[INFO] Total runs planned: {planned_runs}", flush=True)
    print(f"[INFO] Output dir: {run_dir}", flush=True)
    print(f"[INFO] PATH prepended with: {args.ollama_bin_dir.expanduser()}", flush=True)
    print(f"[INFO] Ollama host: {args.ollama_host}", flush=True)
    print(f"[INFO] Ollama binary: {ollama_info['ollama_bin']}", flush=True)
    print(f"[INFO] Ollama log: {ollama_info['ollama_log']}", flush=True)
    if ollama_info["started"]:
        print("[INFO] Auto-started ollama serve in background", flush=True)
    print(f"[INFO] Ollama ready: {ollama_info['ready']}", flush=True)
    if not ollama_info["ready"]:
        print("[WARN] Ollama is not ready. The run may fail unless the service is started manually.", flush=True)

    for doc_idx, target_doc in enumerate(docs, start=1):
        target_tag = slugify_doc(target_doc)
        doc_questions = doc_question_plan.get(target_doc.name, [])
        print(f"[DOC {doc_idx}/{len(docs)}] Build vector DB for {target_doc.name}", flush=True)
        print(f"[DOC {doc_idx}/{len(docs)}] Questions for this doc: {len(doc_questions)}", flush=True)

        if not doc_questions:
            print(f"[DOC {doc_idx}/{len(docs)}] Skip: no aligned questions found", flush=True)
            continue

        vectordb = prepare_data(doc_file=str(target_doc), url_file=None, debug=args.debug)
        qa_chain, retriever, _ = chatbox(vectordb, debug=args.debug, return_prompt=True)

        for q_idx, q in enumerate(doc_questions, start=1):
            print(
                f"[RUN] target={target_tag} q={q_idx}/{len(doc_questions)} "
                f"source={q['source_doc']}",
                flush=True,
            )
            retrieval_start = time.perf_counter()
            retrieved_docs = retriever.invoke(q["question"])
            retrieval_time = time.perf_counter() - retrieval_start

            rendered_prompt = render_prompt(retrieved_docs, q["question"])

            qa_start = time.perf_counter()
            result = qa_chain.invoke({"input": q["question"]})
            qa_time = time.perf_counter() - qa_start
            answer = result["answer"]

            row = {
                "q_global_id": q["q_global_id"],
                "source_doc": q["source_doc"],
                "source_q_id": q["source_q_id"],
                "question": q["question"],
                "target_doc": str(target_doc),
                "target_doc_tag": target_tag,
                "answer": answer,
                "retrieval_time_sec": round(retrieval_time, 4),
                "qa_time_sec": round(qa_time, 4),
                "prompt_template": PROMPT_TEMPLATE_TEXT,
                "prompt_rendered": rendered_prompt,
                "retrieval_docs": serialize_retrieval_docs(retrieved_docs),
            }
            responses.append(row)
            prompt_rows.append(
                {
                    "q_global_id": q["q_global_id"],
                    "source_doc": q["source_doc"],
                    "target_doc_tag": target_tag,
                    "question": q["question"],
                    "prompt_template": PROMPT_TEMPLATE_TEXT,
                    "prompt_rendered": rendered_prompt,
                }
            )

    write_jsonl(run_dir / "responses.jsonl", responses)
    write_jsonl(run_dir / "prompts.jsonl", prompt_rows)

    target_doc_tags = [slugify_doc(doc) for doc in docs]
    write_comparison_csv(run_dir / "comparison.csv", responses, target_doc_tags)

    summary = build_summary(responses, docs, run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(run_dir / "summary.md", summary, docs, question_catalog_path)
    write_analysis_prompt_txt(run_dir / "analysis_prompt.txt", summary, docs, question_catalog_path)

    (run_dir / "question_catalog_snapshot.json").write_text(
        json.dumps(question_catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[INFO] Experiment completed.", flush=True)
    print(f"[INFO] Summary markdown: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Responses jsonl: {run_dir / 'responses.jsonl'}", flush=True)
    print(f"[INFO] AI analysis prompt: {run_dir / 'analysis_prompt.txt'}", flush=True)


if __name__ == "__main__":
    main()



