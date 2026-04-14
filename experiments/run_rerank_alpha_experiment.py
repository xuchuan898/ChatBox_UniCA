import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual-rerank alpha ablation with fixed base retrieval weights."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path (default: auto-detected).",
    )
    parser.add_argument(
        "--doc-file",
        type=Path,
        default=Path("docs/chroma/master.md"),
        help="Single document used in the experiment.",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("questions/questions_batch_example.txt"),
        help="Question file path.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="Rerank alpha values to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k rows to keep in details (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Experiment output base directory.",
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
        help="Log file used when auto-starting ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--auto-start-ollama",
        action="store_true",
        help="Auto-start ollama serve if host is unreachable.",
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
    if os.name == "nt" and not ollama_bin.exists():
        ollama_bin = ollama_bin_dir / "ollama.exe"
    ollama_log = args.ollama_log_file.expanduser().resolve()
    ollama_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_NUM_GPU"] = "1"
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

    if not args.auto_start_ollama or not ollama_bin.exists():
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


def _normalize_question_line(line: str) -> str:
    cleaned = line.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^\s*[-*+]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("#"):
        return ""
    if "?" not in cleaned:
        return ""
    return cleaned


def load_questions(question_file: Path) -> list[str]:
    questions = []
    with question_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            q = _normalize_question_line(raw_line)
            if q:
                questions.append(q)
    return questions


def detect_question_language(text: str) -> str:
    lower = text.lower()
    fr_markers = [
        " le ", " la ", " les ", " des ", " du ", " un ", " une ",
        "dans", "avec", "pour", "est", "responsable", "durée", "courriel",
    ]
    en_markers = [
        " the ", " and ", " for ", " with ", " what ", " which ", " is ", " are ",
        "duration", "internship", "email", "responsible",
    ]
    fr_score = sum(marker in f" {lower} " for marker in fr_markers)
    en_score = sum(marker in f" {lower} " for marker in en_markers)
    if re.search(r"[àâçéèêëîïôûùüÿœ]", lower):
        fr_score += 2
    return "fr" if fr_score >= en_score + 1 else "en"


def extract_retrieval_markdown_blocks(log_text: str) -> list[str]:
    pattern = re.compile(
        r"^\[DEBUG\]\[RETRIEVE_MD_BEGIN\]\n(.*?)\n\[DEBUG\]\[RETRIEVE_MD_END\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_query_variant_events(log_text: str) -> list[list[str]]:
    events = []
    current_variants = []
    in_query = False

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if re.match(r"^\[DEBUG\]\[RETRIEVE\] q=(['\"]).*\1 base_count=\d+ rerank_count=\d+$", line):
            if in_query:
                events.append(current_variants)
            current_variants = []
            in_query = True
            continue

        var_match = re.match(r"^\[DEBUG\]\[RETRIEVE\]\[QUERY_VARIANT\] \d+=(['\"])(.*)\1$", line)
        if var_match and in_query:
            current_variants.append(var_match.group(2))

    if in_query:
        events.append(current_variants)
    return events


def extract_dual_query_events(log_text: str) -> list[dict]:
    events = []
    pattern = re.compile(
        r"^\[DEBUG\]\[RERANK_DUAL\]\[QUERY\] src=(['\"])(.*)\1 translated=(['\"])(.*)\3 alpha=([0-9.]+)$"
    )
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if match:
            events.append(
                {
                    "src_query": match.group(2),
                    "translated_query": match.group(4),
                    "alpha": float(match.group(5)),
                }
            )
    return events


def _parse_table_row(row: str) -> list[str]:
    parts = [p.strip() for p in row.strip().split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def parse_retrieval_block(block: str, top_k: int) -> dict:
    question = ""
    section = ""
    rerank_rows = []

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        q_match = re.match(r"^- Question:\s*`(.*)`\s*$", line)
        if q_match:
            question = q_match.group(1).strip()
            continue
        if line.startswith("#### Rerank Result"):
            section = "rerank"
            continue
        if not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        cells = _parse_table_row(line)
        if not cells or cells[0].lower() == "rank":
            continue
        if section == "rerank" and len(cells) >= 6:
            rerank_rows.append(
                {
                    "rank": int(cells[0]),
                    "score": float(cells[1]),
                    "source": cells[2],
                    "chunk_type": cells[4],
                    "preview": cells[5],
                }
            )

    return {"question": question, "rerank_rows": rerank_rows[:top_k]}


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict]) -> dict:
    top1_scores = [r["rerank_top1_score"] for r in rows if r["rerank_top1_score"] is not None]
    margins = [r["rerank_margin_top1_top2"] for r in rows if r["rerank_margin_top1_top2"] is not None]
    return {
        "questions": len(rows),
        "avg_rerank_top1_score": round(statistics.mean(top1_scores), 4) if top1_scores else 0.0,
        "avg_rerank_margin_top1_top2": round(statistics.mean(margins), 4) if margins else 0.0,
        "unique_top1_chunks": len({r["rerank_top1_preview"] for r in rows if r["rerank_top1_preview"]}),
    }


def build_summary_md(run_dir: Path, summary_rows: list[dict], baseline_alpha: float, top_k: int) -> None:
    lines = [
        "# Rerank Alpha Experiment Summary",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Fixed retrieval weights: `weight_vec=1.25`, `weight_bm25=0.75`, `secondary_variant_weight=0.85`",
        f"- Variant mode: `mapped_current`",
        f"- Rerank details shown per question: Top-{top_k}",
        f"- Baseline alpha: `{baseline_alpha:.2f}`",
        "",
        "| Alpha | Questions | Avg Top1 Score | Avg Margin(1-2) | Unique Top1 Chunks | Top1 Changed vs Baseline |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['questions']} | {row['avg_rerank_top1_score']:.4f} | "
            f"{row['avg_rerank_margin_top1_top2']:.4f} | {row['unique_top1_chunks']} | "
            f"{row['top1_changed_rate_vs_baseline']:.4f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_details_md(run_dir: Path, questions: list[str], run_payloads: list[dict], top_k: int) -> None:
    lines = [
        "# Rerank Alpha Details",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Rerank rows shown per question/alpha: Top-{top_k}",
        "",
    ]
    sorted_runs = sorted(run_payloads, key=lambda r: r["alpha"])

    for q_idx, question in enumerate(questions, start=1):
        lines.append(f"## Q{q_idx}")
        lines.append("")
        lines.append(f"Question: `{_md_escape(question)}`")
        lines.append("")
        for run in sorted_runs:
            retrieval_blocks = run.get("retrieval_blocks", [])
            variant_events = run.get("variant_events", [])
            dual_events = run.get("dual_query_events", [])
            block = retrieval_blocks[q_idx - 1] if q_idx - 1 < len(retrieval_blocks) else {"rerank_rows": []}
            variants = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else []
            dual = dual_events[q_idx - 1] if q_idx - 1 < len(dual_events) else {
                "src_query": question,
                "translated_query": question,
                "alpha": run["alpha"],
            }

            lines.append(f"### alpha={run['alpha']:.2f}")
            lines.append("")
            lines.append(f"- Source query: `{_md_escape(dual.get('src_query', question))}`")
            lines.append(f"- Translated query: `{_md_escape(dual.get('translated_query', question))}`")
            lines.append("")
            lines.append("Variants:")
            if variants:
                for idx, value in enumerate(variants, start=1):
                    lines.append(f"{idx}. `{_md_escape(value)}`")
            else:
                lines.append("1. `_No variant captured_`")
            lines.append("")
            lines.append(f"Rerank Top-{top_k}:")
            lines.append("| Rank | Score | Source | Chunk Type | Preview |")
            lines.append("| ---: | ---: | --- | --- | --- |")
            if block.get("rerank_rows"):
                for row in block["rerank_rows"]:
                    lines.append(
                        f"| {row['rank']} | {row['score']:.4f} | {_md_escape(row['source'])} | "
                        f"{_md_escape(row['chunk_type'])} | {_md_escape(row['preview'])} |"
                    )
            else:
                lines.append("| - | - | - | - | _No rerank rows captured_ |")
            lines.append("")

    (run_dir / "details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    doc_file = resolve_path(project_root, args.doc_file).resolve()
    question_file = resolve_path(project_root, args.question_file).resolve()
    output_base = resolve_path(project_root, args.output_dir).resolve()
    chat_box_path = project_root / "chat_box.py"

    if not doc_file.exists():
        raise FileNotFoundError(f"Doc file not found: {doc_file}")
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not chat_box_path.exists():
        raise FileNotFoundError(f"chat_box.py not found: {chat_box_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    alphas = sorted(set(args.alphas))
    if not alphas:
        raise ValueError("--alphas must not be empty")
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")

    questions = load_questions(question_file)
    if not questions:
        raise ValueError(f"No valid questions in {question_file}")

    ollama_info = ensure_ollama(args)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"rerank_alpha_experiment_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Doc file: {doc_file}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
    print(f"[INFO] Alphas: {alphas}", flush=True)
    print(f"[INFO] Output dir: {run_dir}", flush=True)
    print(f"[INFO] Ollama ready: {ollama_info['ready']}", flush=True)

    run_payloads = []
    question_level_rows = []
    summary_rows = []

    for idx, alpha in enumerate(alphas, start=1):
        alpha_tag = f"a{alpha:.2f}".replace(".", "p")
        log_path = logs_dir / f"{alpha_tag}.log"
        answer_path = answers_dir / f"{alpha_tag}.answers.txt"
        cmd = [
            args.python_bin,
            str(chat_box_path),
            "--doc-file", str(doc_file),
            "--question-file", str(question_file),
            "--answer-file", str(answer_path),
            "--debug",
            "--weight-vec", "1.25",
            "--weight-bm25", "0.75",
            "--secondary-variant-weight", "0.85",
            "--variant-mode", "mapped_current",
            "--rerank-alpha", str(alpha),
        ]

        print(f"[RUN {idx}/{len(alphas)}] alpha={alpha:.2f}", flush=True)
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
        print(f"[RUN {idx}/{len(alphas)}] exit={proc.returncode} duration={duration_sec:.2f}s", flush=True)

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        retrieval_blocks = [parse_retrieval_block(b, args.top_k) for b in extract_retrieval_markdown_blocks(log_text)]
        variant_events = extract_query_variant_events(log_text)
        dual_events = extract_dual_query_events(log_text)

        rows_for_alpha = []
        for q_idx, question in enumerate(questions, start=1):
            lang = detect_question_language(question)
            block = retrieval_blocks[q_idx - 1] if q_idx - 1 < len(retrieval_blocks) else {"rerank_rows": []}
            variants = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else []
            dual = dual_events[q_idx - 1] if q_idx - 1 < len(dual_events) else {
                "src_query": question,
                "translated_query": question,
            }
            rerank_rows = block.get("rerank_rows", [])
            top1 = rerank_rows[0] if rerank_rows else None
            top2 = rerank_rows[1] if len(rerank_rows) > 1 else None
            margin = None
            if top1 and top2:
                margin = round(top1["score"] - top2["score"], 4)

            row = {
                "alpha": alpha,
                "q_id": q_idx,
                "lang": lang,
                "question": question,
                "src_query": dual.get("src_query", question),
                "translated_query": dual.get("translated_query", question),
                "variants": " || ".join(variants),
                "rerank_top1_score": top1["score"] if top1 else None,
                "rerank_top1_source": top1["source"] if top1 else "",
                "rerank_top1_chunk_type": top1["chunk_type"] if top1 else "",
                "rerank_top1_preview": top1["preview"] if top1 else "",
                "rerank_margin_top1_top2": margin,
            }
            rows_for_alpha.append(row)
            question_level_rows.append(row.copy())

        summary = summarize_rows(rows_for_alpha)
        summary.update(
            {
                "alpha": alpha,
                "exit_code": proc.returncode,
                "duration_sec": round(duration_sec, 3),
                "retrieval_blocks": len(retrieval_blocks),
                "questions_expected": len(questions),
            }
        )
        summary_rows.append(summary)

        run_payloads.append(
            {
                "alpha": alpha,
                "exit_code": proc.returncode,
                "duration_sec": round(duration_sec, 3),
                "log_path": str(log_path),
                "answer_path": str(answer_path),
                "retrieval_blocks": retrieval_blocks,
                "variant_events": variant_events,
                "dual_query_events": dual_events,
                "summary": summary,
            }
        )

    baseline_alpha = 1.0 if 1.0 in alphas else alphas[-1]
    baseline_rows = [r for r in question_level_rows if abs(r["alpha"] - baseline_alpha) < 1e-9]
    baseline_by_qid = {r["q_id"]: r for r in baseline_rows}
    for row in question_level_rows:
        if abs(row["alpha"] - baseline_alpha) < 1e-9:
            row["top1_changed_vs_baseline"] = 0
            continue
        base = baseline_by_qid.get(row["q_id"])
        row["top1_changed_vs_baseline"] = 1 if base and row["rerank_top1_preview"] != base["rerank_top1_preview"] else 0

    for summary in summary_rows:
        if abs(summary["alpha"] - baseline_alpha) < 1e-9:
            summary["top1_changed_rate_vs_baseline"] = 0.0
            continue
        rows = [r for r in question_level_rows if abs(r["alpha"] - summary["alpha"]) < 1e-9]
        changed = [r["top1_changed_vs_baseline"] for r in rows]
        summary["top1_changed_rate_vs_baseline"] = round(sum(changed) / len(changed), 4) if changed else 0.0

    write_csv(
        run_dir / "alpha_summary.csv",
        summary_rows,
        [
            "alpha",
            "exit_code",
            "duration_sec",
            "questions_expected",
            "retrieval_blocks",
            "questions",
            "avg_rerank_top1_score",
            "avg_rerank_margin_top1_top2",
            "unique_top1_chunks",
            "top1_changed_rate_vs_baseline",
        ],
    )

    write_csv(
        run_dir / "alpha_question_level.csv",
        question_level_rows,
        [
            "alpha",
            "q_id",
            "lang",
            "question",
            "src_query",
            "translated_query",
            "variants",
            "rerank_top1_score",
            "rerank_top1_source",
            "rerank_top1_chunk_type",
            "rerank_top1_preview",
            "rerank_margin_top1_top2",
            "top1_changed_vs_baseline",
        ],
    )

    (run_dir / "runs.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc_file": str(doc_file),
                "question_file": str(question_file),
                "top_k": args.top_k,
                "fixed_weights": {
                    "weight_vec": 1.25,
                    "weight_bm25": 0.75,
                    "secondary_variant_weight": 0.85,
                    "variant_mode": "mapped_current",
                },
                "baseline_alpha": baseline_alpha,
                "runs": run_payloads,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    build_summary_md(run_dir, summary_rows, baseline_alpha, args.top_k)
    build_details_md(run_dir, questions, run_payloads, args.top_k)

    print("[INFO] Alpha experiment finished.", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Details: {run_dir / 'details.md'}", flush=True)
    print(f"[INFO] Alpha summary CSV: {run_dir / 'alpha_summary.csv'}", flush=True)
    print(f"[INFO] Question-level CSV: {run_dir / 'alpha_question_level.csv'}", flush=True)


if __name__ == "__main__":
    main()
