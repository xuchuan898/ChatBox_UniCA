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


def slug_for_doc(doc_path: Path) -> str:
    return f"{doc_path.stem}_{doc_path.suffix.lstrip('.').lower()}"


def extract_metric_line(log_text: str, pattern: str) -> str:
    match = re.search(pattern, log_text, re.MULTILINE)
    return match.group(0) if match else ""


def extract_all_metric_lines(log_text: str, pattern: str) -> list[str]:
    return re.findall(pattern, log_text, re.MULTILINE)


def parse_answers(answer_file: Path) -> list[dict]:
    rows = []
    if not answer_file.exists():
        return rows

    lines = answer_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        q_match = re.match(r"^\[Q(\d+)\]\s*(.*)$", lines[i].strip())
        if not q_match:
            i += 1
            continue
        q_id = int(q_match.group(1))
        question = q_match.group(2).strip()
        answer = ""
        if i + 1 < len(lines):
            a_match = re.match(r"^\[A\d+\]\s*(.*)$", lines[i + 1].strip())
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
        if run["qa_time_lines"]:
            lines.append("- QA invoke times:")
            for ql in run["qa_time_lines"]:
                lines.append(f"  - {ql}")
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
        print(f"[RUN {idx}/{len(docs)}] Start: {tag}", flush=True)
        print(f"[RUN {idx}/{len(docs)}] Doc: {doc}", flush=True)
        print(f"[RUN {idx}/{len(docs)}] Log: {log_path}", flush=True)

        if not doc.exists():
            log_path.write_text(f"Missing doc: {doc}\n", encoding="utf-8")
            print(f"[RUN {idx}/{len(docs)}] Skip missing doc", flush=True)
            runs.append(
                {
                    "tag": tag,
                    "exit_code": 404,
                    "duration_sec": 0.0,
                    "doc_path": str(doc),
                    "log_path": str(log_path),
                    "answer_path": str(answer_path),
                    "loaded_line": "",
                    "chunks_line": "",
                    "embedding_line": "",
                    "prepare_line": "",
                    "qa_time_lines": [],
                    "answers": [],
                }
            )
            continue

        cmd = [
            args.python_bin,
            str(chat_box_path),
            "--doc-file",
            str(doc),
            "--question-file",
            str(question_file),
            "--answer-file",
            str(answer_path),
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
        runs.append(
            {
                "tag": tag,
                "exit_code": proc.returncode,
                "duration_sec": duration_sec,
                "doc_path": str(doc),
                "log_path": str(log_path),
                "answer_path": str(answer_path),
                "loaded_line": extract_metric_line(log_text, r"^\[DEBUG\] Loaded documents:.*$"),
                "chunks_line": extract_metric_line(log_text, r"^\[DEBUG\] Total chunks after split:.*$"),
                "embedding_line": extract_metric_line(log_text, r"^\[DEBUG\] Embedding \+ vector DB build time:.*$"),
                "prepare_line": extract_metric_line(log_text, r"^\[DEBUG\] prepare_data total time:.*$"),
                "qa_time_lines": extract_all_metric_lines(log_text, r"^\[DEBUG\] QA invoke time:.*$"),
                "error_hint": error_hint,
                "answers": parse_answers(answer_path),
            }
        )

    write_summary_md(run_dir, question_file, runs)
    write_comparison_csv(run_dir, runs)
    (run_dir / "summary.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Experiment done.", flush=True)
    print(f"[INFO] Output directory: {run_dir}", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Comparison CSV: {run_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
