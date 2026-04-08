import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch QA experiment across master.{md,txt,pdf} and collect comparable outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path (default: auto-detected from this script location).",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("questions/questions_batch_example.txt"),
        help="Batch question file path (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        nargs="*",
        default=[
            Path("docs/chroma/master.md"),
            Path("docs/chroma/master.txt"),
            Path("docs/chroma/master.pdf"),
        ],
        help="Doc files to test (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Experiment results base directory (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run chat_box.py.",
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
    ollama_log = args.ollama_log_file.expanduser().resolve()
    ollama_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    # Force ollama to use only GPU 0 to avoid multi‑GPU crashes
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_NUM_GPU"] = "1"

    info = {
        "env": env,
        "ollama_bin": str(ollama_bin),
        "ollama_log": str(ollama_log),
        "started": False,
        "ready": False,
        "model_pulled": False,
    }

    if ollama_is_ready(args.ollama_host):
        info["ready"] = True
        try:
            result = subprocess.run([
                str(ollama_bin), "list"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "gemma3:1b" not in result.stdout:
                pull_result = subprocess.run([
                    str(ollama_bin), "pull", "gemma3:1b"
                ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                info["model_pulled"] = pull_result.returncode == 0
        except Exception as e:
            info["model_pulled"] = False
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

    if info["ready"]:
        try:
            result = subprocess.run([
                str(ollama_bin), "list"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "gemma3:1b" not in result.stdout:
                pull_result = subprocess.run([
                    str(ollama_bin), "pull", "gemma3:1b"
                ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                info["model_pulled"] = pull_result.returncode == 0
        except Exception as e:
            info["model_pulled"] = False

    return info


def slug_for_doc(doc_path: Path) -> str:
    return f"{doc_path.stem}_{doc_path.suffix.lstrip('.').lower()}"


def extract_metric_line(log_text: str, pattern: str) -> str:
    match = re.search(pattern, log_text, re.MULTILINE)
    return match.group(0) if match else ""


def extract_all_metric_lines(log_text: str, pattern: str) -> list[str]:
    return re.findall(pattern, log_text, re.MULTILINE)


def extract_qa_times(log_text: str) -> list[float]:
    values = re.findall(r"^\[DEBUG] QA invoke time:\s*([0-9]+(?:\.[0-9]+)?)s$", log_text, re.MULTILINE)
    return [float(v) for v in values]


def extract_retrieval_markdown_blocks(log_text: str) -> list[str]:
    pattern = re.compile(
        r"^\\[DEBUG\\]\\[RETRIEVE_MD_BEGIN\\]\\n(.*?)\\n\\[DEBUG\\]\\[RETRIEVE_MD_END\\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_prompt_markdown_blocks(log_text: str) -> list[str]:
    pattern = re.compile(
        r"^\\[DEBUG\\]\\[PROMPT_MD_BEGIN\\]\\n(.*?)\\n\\[DEBUG\\]\\[PROMPT_MD_END\\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_retrieval_events(log_text: str) -> list[dict]:
    events = []
    current = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        q_match = re.match(
            r"^\\[DEBUG\\]\\[RETRIEVE\\] q=(['\"])(.*)\\1 base_count=(\\d+) rerank_count=(\\d+)$",
            line,
        )
        if q_match:
            if current:
                events.append(current)
            current = {
                "question": q_match.group(2),
                "base_count": int(q_match.group(3)),
                "rerank_count": int(q_match.group(4)),
                "base_stats": "",
                "rerank_stats": "",
                "base_top": [],
                "rerank_top": [],
            }
            continue

        if not current:
            continue

        base_stats_match = re.match(r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[BASE_STATS\\] (.*)$", line)
        if base_stats_match:
            current["base_stats"] = base_stats_match.group(1)
            continue

        rerank_stats_match = re.match(r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[RERANK_STATS\\] (.*)$", line)
        if rerank_stats_match:
            current["rerank_stats"] = rerank_stats_match.group(1)
            continue

        base_top_match = re.match(
            r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[BASE_TOP\\] rank=(\\d+) score=([0-9]+(?:\\.[0-9]+)?) source=(.*)$",
            line,
        )
        if base_top_match:
            if current["base_top"] is not None:
                current["base_top"].append(
                    {
                        "rank": int(base_top_match.group(1)),
                        "score": float(base_top_match.group(2)),
                        "source": base_top_match.group(3).strip(),
                    }
                )
            continue

        rerank_top_match = re.match(
            r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[RERANK_TOP\\] rank=(\\d+) score=([0-9]+(?:\\.[0-9]+)?) source=(.*)$",
            line,
        )
        if rerank_top_match:
            if current["rerank_top"] is not None:
                current["rerank_top"].append(
                    {
                        "rank": int(rerank_top_match.group(1)),
                        "score": float(rerank_top_match.group(2)),
                        "source": rerank_top_match.group(3).strip(),
                    }
                )
            continue

    if current:
        events.append(current)

    return events


def parse_retrieval_markdown_block(block: str) -> dict:
    question = ""
    base_candidates = None
    reranked_kept = None
    cleaned_lines = []
    previous_nonempty = ""

    for raw_line in block.splitlines():
        line = raw_line.rstrip()

        q_match = re.match(r"^- Question:\s*`(.*)`\s*$", line)
        if q_match:
            question = q_match.group(1).strip()
            continue

        base_match = re.match(r"^- Base candidates:\s*(\d+)\s*$", line)
        if base_match:
            base_candidates = int(base_match.group(1))
            continue

        rerank_match = re.match(r"^- Reranked kept:\s*(\d+)\s*$", line)
        if rerank_match:
            reranked_kept = int(rerank_match.group(1))
            continue

        if line.strip() == "### Retrieval Trace":
            continue

        if line.strip() and line.strip() == previous_nonempty:
            continue

        cleaned_lines.append(line)
        if line.strip():
            previous_nonempty = line.strip()

    cleaned_text = "\n".join(cleaned_lines).strip()
    return {
        "question": question,
        "base_candidates": base_candidates,
        "reranked_kept": reranked_kept,
        "cleaned_text": cleaned_text,
    }


def render_retrieval_fallback(event: dict) -> str:
    lines = ["#### Retrieval Trace (fallback from debug lines)", ""]
    lines.append(f"- Question: `{event.get('question', '')}`")
    lines.append(f"- Base candidates: {event.get('base_count', 0)}")
    lines.append(f"- Reranked kept: {event.get('rerank_count', 0)}")

    if event.get("base_stats"):
        lines.append(f"- Base stats: {event['base_stats']}")
    if event.get("rerank_stats"):
        lines.append(f"- Rerank stats: {event['rerank_stats']}")

    base_top = event.get("base_top", [])
    if base_top:
        lines.append("")
        lines.append("#### Base Top")
        lines.append("| Rank | Score | Source |")
        lines.append("| ---: | ---: | --- |")
        for item in base_top[:5]:
            lines.append(f"| {item['rank']} | {item['score']:.4f} | {item['source']} |")

    rerank_top = event.get("rerank_top", [])
    if rerank_top:
        lines.append("")
        lines.append("#### Rerank Top")
        lines.append("| Rank | Score | Source |")
        lines.append("| ---: | ---: | --- |")
        for item in rerank_top[:5]:
            lines.append(f"| {item['rank']} | {item['score']:.4f} | {item['source']} |")

    return "\n".join(lines)


def parse_answers(answer_file: Path) -> list[dict]:
    rows = []
    if not answer_file.exists():
        return rows

    lines = answer_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        q_match = re.match(r"^\[Q(\d+)]\s*(.*)$", lines[i].strip())
        if not q_match:
            i += 1
            continue
        q_id = int(q_match.group(1))
        question = q_match.group(2).strip()
        answer = ""
        if i + 1 < len(lines):
            a_match = re.match(r"^\[A\d+]\s*(.*)$", lines[i + 1].strip())
            if a_match:
                answer = a_match.group(1).strip()
        rows.append({"q_id": q_id, "question": question, "answer": answer})
        i += 1
    return rows


def write_summary_md(run_dir: Path, question_file: Path, runs: list[dict]) -> None:
    summary_path = run_dir / "summary.md"
    lines = []
    lines.append("# Format Experiment Summary")
    lines.append("")
    lines.append(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Question file: {question_file}")
    lines.append("")
    lines.append("## Run Table")
    lines.append("")
    lines.append("| Tag | Exit Code | Duration(s) | Doc | Log | Answers |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for run in runs:
        lines.append(
            f"| {run['tag']} | {run['exit_code']} | {run['duration_sec']:.2f} | "
            f"{run['doc_path']} | {run['log_path']} | {run['answer_path']} |"
        )

    lines.append("")
    lines.append("## Key Debug Metrics")
    lines.append("")
    for run in runs:
        lines.append(f"### {run['tag']}")
        lines.append("")
        if run["error_hint"]:
            lines.append(f"- Error hint: {run['error_hint']}")
        if run["loaded_line"]:
            lines.append(f"- {run['loaded_line']}")
        if run["chunks_line"]:
            lines.append(f"- {run['chunks_line']}")
        if run["embedding_line"]:
            lines.append(f"- {run['embedding_line']}")
        if run["prepare_line"]:
            lines.append(f"- {run['prepare_line']}")
        qa_times = run.get("qa_times_sec", [])
        if qa_times:
            lines.append(
                "- QA time(s): "
                f"count={len(qa_times)}, min={min(qa_times):.2f}, "
                f"avg={sum(qa_times) / len(qa_times):.2f}, max={max(qa_times):.2f}"
            )
        retrieval_calls = run.get("retrieval_calls", 0)
        if retrieval_calls:
            lines.append(f"- Retrieval calls: {retrieval_calls}")
        if run.get("retrieval_base_stats"):
            lines.append("- Retrieval base score stats (first 3):")
            for line in run["retrieval_base_stats"][:3]:
                lines.append(f"  - {line}")
        if run.get("retrieval_rerank_stats"):
            lines.append("- Retrieval rerank score stats (first 3):")
            for line in run["retrieval_rerank_stats"][:3]:
                lines.append(f"  - {line}")
        if run.get("retrieval_base_top"):
            lines.append("- Retrieval base top docs (first 3 lines):")
            for line in run["retrieval_base_top"][:3]:
                lines.append(f"  - {line}")
        if run.get("retrieval_rerank_top"):
            lines.append("- Retrieval rerank top docs (first 3 lines):")
            for line in run["retrieval_rerank_top"][:3]:
                lines.append(f"  - {line}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Use `comparison.csv` for side-by-side answer review.")
    lines.append("- Use raw logs to inspect retrieval details and failure points.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_csv(run_dir: Path, runs: list[dict]) -> None:
    all_by_qid = {}
    tags = [run["tag"] for run in runs]

    for run in runs:
        for item in run["answers"]:
            q_id = item["q_id"]
            if q_id not in all_by_qid:
                all_by_qid[q_id] = {"q_id": q_id, "question": item["question"]}
            all_by_qid[q_id][f"answer__{run['tag']}"] = item["answer"]

    output_path = run_dir / "comparison.csv"
    fieldnames = ["q_id", "question"] + [f"answer__{tag}" for tag in tags]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q_id in sorted(all_by_qid.keys()):
            row = all_by_qid[q_id]
            for tag in tags:
                row.setdefault(f"answer__{tag}", "")
            writer.writerow(row)


def write_retrieval_trace_md(run_dir: Path, runs: list[dict]) -> None:
    output_path = run_dir / "retrieval_trace.md"
    lines = [
        "# Retrieval Trace",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Source: debug retrieval markdown blocks and retrieval debug lines",
        "- Layout: grouped by question, then by target doc tag",
        "",
    ]

    question_map = {}
    for run in runs:
        answers = run.get("answers", [])
        retrieval_blocks = run.get("retrieval_md_blocks", [])
        prompt_blocks = run.get("prompt_md_blocks", [])
        retrieval_events = run.get("retrieval_events", [])
        parsed_blocks = [parse_retrieval_markdown_block(block) for block in retrieval_blocks]
        item_count = max(len(answers), len(parsed_blocks), len(prompt_blocks), len(retrieval_events))

        for idx in range(item_count):
            answer_item = answers[idx] if idx < len(answers) else {}
            block_item = parsed_blocks[idx] if idx < len(parsed_blocks) else {}
            event_item = retrieval_events[idx] if idx < len(retrieval_events) else {}

            q_id = answer_item.get("q_id", idx + 1)
            question = (
                answer_item.get("question")
                or block_item.get("question")
                or event_item.get("question")
                or "(question unavailable)"
            )

            if q_id not in question_map:
                question_map[q_id] = {"question": question, "items": []}

            question_map[q_id]["items"].append(
                {
                    "tag": run["tag"],
                    "question": question,
                    "prompt": prompt_blocks[idx] if idx < len(prompt_blocks) else "",
                    "block": block_item,
                    "event": event_item,
                }
            )

    for q_id in sorted(question_map.keys()):
        q_entry = question_map[q_id]
        lines.append(f"## Q{q_id}")
        lines.append("")
        lines.append(f"- Question: `{q_entry['question']}`")
        lines.append("")
        lines.append("| Target Doc | Base Candidates | Reranked Kept | Retrieval Source | Prompt |")
        lines.append("| --- | ---: | ---: | --- | --- |")

        for item in q_entry["items"]:
            block = item.get("block", {})
            event = item.get("event", {})
            base_candidates = block.get("base_candidates")
            reranked_kept = block.get("reranked_kept")

            if base_candidates is None:
                base_candidates = event.get("base_count", "-")
            if reranked_kept is None:
                reranked_kept = event.get("rerank_count", "-")

            source_label = "markdown block" if block.get("cleaned_text") else ("debug fallback" if event else "missing")
            prompt_label = "yes" if item.get("prompt") else "no"
            lines.append(
                f"| {item['tag']} | {base_candidates} | {reranked_kept} | {source_label} | {prompt_label} |"
            )

        lines.append("")

        for item in q_entry["items"]:
            lines.append(f"### {item['tag']}")
            lines.append("")

            # Prompt details block
            prompt_text = item.get("prompt", "").strip()
            if prompt_text:
                lines.append("<details>")
                lines.append("<summary>Prompt sent to LLM</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(prompt_text)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")

            # Retrieval details block
            block = item.get("block", {})
            retrieval_detail = block.get("cleaned_text", "").strip()
            if not retrieval_detail:
                event = item.get("event", {})
                retrieval_detail = render_retrieval_fallback(event) if event else "_No retrieval trace captured._"

            lines.append("<details>")
            lines.append("<summary>Retrieval details</summary>")
            lines.append("")
            lines.append(retrieval_detail)
            lines.append("")
            lines.append("</details>")
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    question_file = resolve_path(project_root, args.question_file).resolve()
    docs = [resolve_path(project_root, d).resolve() for d in args.docs]
    output_base = resolve_path(project_root, args.output_dir).resolve()
    ollama_info = ensure_ollama(args)

    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not docs:
        raise ValueError("No doc files provided.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"format_compare_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Project root: {project_root}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
    print(f"[INFO] Output dir: {run_dir}", flush=True)
    print(f"[INFO] Total docs to test: {len(docs)}", flush=True)
    print(f"[INFO] PATH prepended with: {args.ollama_bin_dir.expanduser()}", flush=True)
    print(f"[INFO] Ollama host: {args.ollama_host}", flush=True)
    print(f"[INFO] Ollama binary: {ollama_info['ollama_bin']}", flush=True)
    print(f"[INFO] Ollama log: {ollama_info['ollama_log']}", flush=True)
    if ollama_info["started"]:
        print("[INFO] Auto-started ollama serve in background", flush=True)
    print(f"[INFO] Ollama ready: {ollama_info['ready']}", flush=True)

    runs = []
    chat_box_path = project_root / "chat_box.py"
    for idx, doc in enumerate(docs, start=1):
        tag = slug_for_doc(doc)
        log_path = logs_dir / f"{tag}.log"
        answer_path = answers_dir / f"{tag}.answers.txt"
        chunk_path = answers_dir / f"{tag}.chunks.jsonl"  # Always save chunk info
        print(f"[RUN {idx}/{len(docs)}] Start: {tag}", flush=True)
        print(f"[RUN {idx}/{len(docs)}] Doc: {doc}", flush=True)
        print(f"[RUN {idx}/{len(docs)}] Log: {log_path}", flush=True)
        print(f"[RUN {idx}/{len(docs)}] Chunks file: {chunk_path}", flush=True)

        if not doc.exists():
            log_path.write_text(f"Missing doc: {doc}\n", encoding="utf-8")
            print(f"[RUN {idx}/{len(docs)}] Skip missing doc", flush=True)
            runs.append({
                "tag": tag,
                "exit_code": 404,
                "duration_sec": 0.0,
                "doc_path": str(doc),
                "log_path": str(log_path),
                "answer_path": str(answer_path),
                "chunk_path": str(chunk_path),
                "loaded_line": "",
                "chunks_line": "",
                "embedding_line": "",
                "prepare_line": "",
                "qa_time_lines": [],
                "qa_times_sec": [],
                "retrieval_calls": 0,
                "retrieval_base_stats": [],
                "retrieval_rerank_stats": [],
                "retrieval_base_top": [],
                "retrieval_rerank_top": [],
                "retrieval_md_blocks": [],
                "prompt_md_blocks": [],
                "retrieval_events": [],
                "answers": [],
                "chunk_summary": "",
                "error_hint": "",
            })
            continue

        cmd = [
            args.python_bin,
            str(chat_box_path),
            "--doc-file", str(doc),
            "--question-file", str(question_file),
            "--answer-file", str(answer_path),
            "--save-chunks-file", str(chunk_path),  # Always pass chunk file
            "--debug",
        ]
        print(f"[RUN {idx}/{len(docs)}] Command: {' '.join(cmd)}", flush=True)

        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                env=ollama_info["env"],
            )
        duration_sec = time.perf_counter() - start
        print(
            f"[RUN {idx}/{len(docs)}] Finished in {duration_sec:.2f}s with exit code {proc.returncode}",
            flush=True,
        )

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        error_hint = ""
        for line in log_text.splitlines():
            if "Traceback" in line:
                continue
            if "Error" in line or "Exception" in line or "ModuleNotFoundError" in line:
                error_hint = line.strip()
                break
        if error_hint:
            print(f"[RUN {idx}/{len(docs)}] Error hint: {error_hint}", flush=True)
        else:
            print(f"[RUN {idx}/{len(docs)}] No immediate error hint detected", flush=True)

        # Summarize chunk info
        chunk_summary = ""
        chunk_count = 0
        chunk_types = {}
        chunk_previews = []
        if chunk_path.exists():
            try:
                with chunk_path.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        try:
                            rec = json.loads(line)
                            chunk_count += 1
                            ctype = rec.get("chunk_type", "unknown")
                            chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
                            if i < 3:
                                chunk_previews.append(rec["content"][:80].replace("\n", " "))
                        except Exception:
                            continue
                chunk_summary = f"chunks={chunk_count}, types={chunk_types}, previews={chunk_previews}"
            except Exception as e:
                chunk_summary = f"[ERROR reading chunk file: {e}]"
        else:
            chunk_summary = "[No chunk file generated]"

        runs.append({
            "tag": tag,
            "exit_code": proc.returncode,
            "duration_sec": duration_sec,
            "doc_path": str(doc),
            "log_path": str(log_path),
            "answer_path": str(answer_path),
            "chunk_path": str(chunk_path),
            "loaded_line": extract_metric_line(log_text, r"^\\[DEBUG\\] Loaded documents:.*$"),
            "chunks_line": extract_metric_line(log_text, r"^\\[DEBUG\\] Total chunks after split:.*$"),
            "embedding_line": extract_metric_line(log_text, r"^\\[DEBUG\\] Embedding \\+ vector DB build time:.*$"),
            "prepare_line": extract_metric_line(log_text, r"^\\[DEBUG\\] prepare_data total time:.*$"),
            "qa_time_lines": extract_all_metric_lines(log_text, r"^\\[DEBUG\\] QA invoke time:.*$"),
            "qa_times_sec": extract_qa_times(log_text),
            "retrieval_calls": len(extract_all_metric_lines(log_text, r"^\\[DEBUG\\]\\[RETRIEVE\\] q=.*$")),
            "retrieval_base_stats": extract_all_metric_lines(log_text, r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[BASE_STATS\\] .*$"),
            "retrieval_rerank_stats": extract_all_metric_lines(log_text, r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[RERANK_STATS\\] .*$"),
            "retrieval_base_top": extract_all_metric_lines(log_text, r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[BASE_TOP\\] .*$"),
            "retrieval_rerank_top": extract_all_metric_lines(log_text, r"^\\[DEBUG\\]\\[RETRIEVE\\]\\[RERANK_TOP\\] .*$"),
            "retrieval_md_blocks": extract_retrieval_markdown_blocks(log_text),
            "prompt_md_blocks": extract_prompt_markdown_blocks(log_text),
            "retrieval_events": extract_retrieval_events(log_text),
            "answers": parse_answers(answer_path),
            "chunk_summary": chunk_summary,
            "error_hint": error_hint,
        })

    write_summary_md(run_dir, question_file, runs)
    write_comparison_csv(run_dir, runs)
    write_retrieval_trace_md(run_dir, runs)
    (run_dir / "summary.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Experiment done.", flush=True)
    print(f"[INFO] Output directory: {run_dir}", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Comparison CSV: {run_dir / 'comparison.csv'}", flush=True)
    print(f"[INFO] Retrieval Trace: {run_dir / 'retrieval_trace.md'}", flush=True)


if __name__ == "__main__":
    main()
