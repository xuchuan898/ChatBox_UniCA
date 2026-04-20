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
        default=Path("questions/questions_batch_student_short_typo_en_fr.txt"),
        help="Batch question file path (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("questions/questions_batch_student_short_typo_en_fr_gold.json"),
        help="Gold file with expected answers and gold chunks for retrieval evaluation.",
    )
    parser.add_argument(
        "--gold-chunks-file",
        type=Path,
        default=None,
        help=(
            "Reference chunks jsonl used for strict retrieval evaluation by chunk id. "
            "If omitted, uses current run output: answers/master_md.chunks.jsonl."
        ),
    )
    parser.add_argument(
        "--eval-topk",
        type=str,
        default="1,5,8",
        help="Comma-separated top-k values for retrieval evaluation (e.g., 1,5,8).",
    )
    parser.add_argument(
        "--dynamic-topk-ratio",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9",
        help=(
            "Comma-separated dynamic top-k ratios. "
            "For each ratio r, keep candidates with score >= r * max_score."
        ),
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


def parse_eval_topks(value: str) -> list[int]:
    topks = []
    for token in (value or "").split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError(f"Invalid top-k value: {k}. Must be > 0.")
        topks.append(k)
    unique_sorted = sorted(set(topks))
    if not unique_sorted:
        raise ValueError("No valid top-k values provided.")
    return unique_sorted


def parse_dynamic_topk_ratios(value: str) -> list[float]:
    ratios = []
    for token in (value or "").split(","):
        token = token.strip()
        if not token:
            continue
        ratio = float(token)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Invalid dynamic top-k ratio: {ratio}. Must be in (0, 1].")
        ratios.append(round(ratio, 4))
    unique_sorted = sorted(set(ratios))
    if not unique_sorted:
        raise ValueError("No valid dynamic top-k ratios provided.")
    return unique_sorted


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
        r"^\[DEBUG\]\[RETRIEVE_MD_BEGIN\]\n(.*?)\n\[DEBUG\]\[RETRIEVE_MD_END\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_prompt_markdown_blocks(log_text: str) -> list[str]:
    pattern = re.compile(
        r"^\[DEBUG\]\[PROMPT_MD_BEGIN\]\n(.*?)\n\[DEBUG\]\[PROMPT_MD_END\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_retrieval_events(log_text: str) -> list[dict]:
    events = []
    current = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        q_match = re.match(
            r"^\[DEBUG\]\[RETRIEVE\] q=(['\"])(.*)\1 base_count=(\d+) rerank_count=(\d+)$",
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

        base_stats_match = re.match(r"^\[DEBUG\]\[RETRIEVE\]\[BASE_STATS\] (.*)$", line)
        if base_stats_match:
            current["base_stats"] = base_stats_match.group(1)
            continue

        rerank_stats_match = re.match(r"^\[DEBUG\]\[RETRIEVE\]\[RERANK_STATS\] (.*)$", line)
        if rerank_stats_match:
            current["rerank_stats"] = rerank_stats_match.group(1)
            continue

        base_top_match = re.match(
            r"^\[DEBUG\]\[RETRIEVE\]\[BASE_TOP\] rank=(\d+) score=([0-9]+(?:\.[0-9]+)?) source=(.*)$",
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
            r"^\[DEBUG\]\[RETRIEVE\]\[RERANK_TOP\] rank=(\d+) score=([0-9]+(?:\.[0-9]+)?) source=(.*)$",
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


def render_retrieval_fallback(event: dict, max_items: int | None = 5) -> str:
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
        base_items = base_top if max_items is None else base_top[:max_items]
        for item in base_items:
            lines.append(f"| {item['rank']} | {item['score']:.4f} | {item['source']} |")

    rerank_top = event.get("rerank_top", [])
    if rerank_top:
        lines.append("")
        lines.append("#### Rerank Top")
        lines.append("| Rank | Score | Source |")
        lines.append("| ---: | ---: | --- |")
        rerank_items = rerank_top if max_items is None else rerank_top[:max_items]
        for item in rerank_items:
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


def load_gold_map(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for item in payload.get("items", []):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        chunk_ids = item.get("gold_chunk_ids")
        if chunk_ids is None:
            chunk_id = item.get("gold_chunk_id")
            chunk_ids = [chunk_id] if chunk_id is not None else []
        normalized_chunk_ids = []
        for cid in chunk_ids:
            try:
                normalized_chunk_ids.append(int(cid))
            except (TypeError, ValueError):
                continue
        mapping[question] = {
            "expected_answer": item.get("expected_answer", ""),
            "gold_chunk": item.get("gold_chunk", ""),
            "gold_chunk_ids": normalized_chunk_ids,
            "lang": item.get("lang", ""),
            "qid": item.get("qid", ""),
        }
    return mapping


def _normalize_for_match(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿœ\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def extract_context_label(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\[Context:\s*(.*?)\]", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def load_chunk_catalog(path: Path | None) -> dict:
    catalog = {
        "path": str(path) if path else "",
        "chunks": [],
        "by_context": {},
    }
    if not path or not path.exists():
        return catalog

    chunks = []
    by_context = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            content = str(rec.get("content", ""))
            context = extract_context_label(content)
            norm_content = _normalize_for_match(content)
            norm_context = _normalize_for_match(context)
            item = {
                "chunk_id": idx,
                "content": content,
                "norm_content": norm_content,
                "context": context,
                "norm_context": norm_context,
                "chunk_type": rec.get("chunk_type", ""),
            }
            chunks.append(item)
            if norm_context:
                by_context.setdefault(norm_context, []).append(idx)

    catalog["chunks"] = chunks
    catalog["by_context"] = by_context
    return catalog


def match_preview_to_chunk_ids(preview: str, chunk_catalog: dict) -> list[int]:
    if not preview or not chunk_catalog.get("chunks"):
        return []

    context = extract_context_label(preview)
    norm_context = _normalize_for_match(context)
    norm_preview = _normalize_for_match(preview.replace("...", " "))
    if len(norm_preview) < 8:
        return []

    chunks = chunk_catalog.get("chunks", [])
    by_context = chunk_catalog.get("by_context", {})

    # Prefer context matching when available.
    if norm_context:
        context_ids = by_context.get(norm_context, [])
        if len(context_ids) == 1:
            return context_ids
        if context_ids:
            narrowed = [cid for cid in context_ids if norm_preview in chunks[cid]["norm_content"]]
            if len(narrowed) == 1:
                return narrowed
            if narrowed:
                return sorted(set(narrowed))

    # Fallback: normalized substring match over all chunks.
    matched = [c["chunk_id"] for c in chunks if norm_preview in c["norm_content"]]
    return sorted(set(matched))


def parse_rerank_rows_from_block(block: str) -> dict:
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
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [part.strip() for part in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
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
    return {"question": question, "rerank_rows": rerank_rows}


def write_summary_md(
    run_dir: Path,
    question_file: Path,
    runs: list[dict],
    eval_topks: list[int],
    dynamic_topk_ratios: list[float],
) -> None:
    summary_path = run_dir / "summary.md"
    lines = []
    lines.append("# Format Experiment Summary")
    lines.append("")
    lines.append(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Question file: {question_file}")
    ratio_labels = ", ".join(f"{r:.2f}" for r in dynamic_topk_ratios)
    lines.append(f"- Dynamic top-k ratios: [{ratio_labels}]")
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
        hit_counts = run.get("retrieval_hit_counts", {})
        hit_rates = run.get("retrieval_hit_rates", {})
        total_eval = run.get("retrieval_eval_total", 0)
        if total_eval:
            for k in eval_topks:
                count = hit_counts.get(str(k), 0)
                rate = hit_rates.get(str(k))
                rate_txt = f"{rate:.3f}" if rate is not None else "n/a"
                lines.append(f"- Retrieval Hit@{k}: {count}/{total_eval} ({rate_txt})")
        dyn_by_ratio = run.get("retrieval_dynamic_by_ratio", {})
        for ratio in dynamic_topk_ratios:
            key = f"{ratio:.4f}"
            stat = dyn_by_ratio.get(key, {})
            dyn_count = stat.get("hit_count", 0)
            dyn_rate = stat.get("hit_rate")
            dyn_k_avg = stat.get("avg_k")
            if total_eval and dyn_rate is not None:
                dyn_rate_txt = f"{dyn_rate:.3f}"
                dyn_k_avg_txt = f"{dyn_k_avg:.2f}" if dyn_k_avg is not None else "n/a"
                lines.append(
                    f"- Retrieval Hit@DynamicK(r={ratio:.2f}): "
                    f"{dyn_count}/{total_eval} ({dyn_rate_txt}), avg K={dyn_k_avg_txt}"
                )
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
    lines.append("- Use `retrieval_eval.md` and `retrieval_eval.csv` for retrieval-focused evaluation by strict chunk-id hit@k and dynamic-k.")
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


def write_retrieval_eval_csv(
    run_dir: Path,
    runs: list[dict],
    eval_topks: list[int],
    dynamic_topk_ratios: list[float],
) -> None:
    rows = []
    for run in runs:
        for item in run.get("retrieval_eval", []):
            rows.append(
                {
                    "tag": run["tag"],
                    "doc_path": run["doc_path"],
                    **item,
                }
            )

    output_path = run_dir / "retrieval_eval.csv"
    fieldnames = [
        "tag",
        "doc_path",
        "q_id",
        "question",
        "gold_chunk_ids",
        "gold_chunk",
        "expected_answer",
        "top1_matched_chunk_ids",
        "top1_score",
        "top1_preview",
    ]
    for k in eval_topks:
        fieldnames.append(f"hit_at_{k}")
    fieldnames.extend(["dynamic_ratio", "dynamic_k", "hit_at_dynamic_k"])
    fieldnames.append("dynamic_by_ratio")
    for ratio in dynamic_topk_ratios:
        ratio_slug = f"{ratio:.2f}".replace(".", "_")
        fieldnames.extend([f"dynamic_k_r{ratio_slug}", f"hit_at_dynamic_k_r{ratio_slug}"])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            first_ratio = dynamic_topk_ratios[0]
            first_key = f"{first_ratio:.4f}"
            dyn_map = row.get("dynamic_by_ratio", {})
            first_dyn = dyn_map.get(first_key, {})
            row.setdefault("dynamic_ratio", f"{first_ratio:.4f}")
            row.setdefault("dynamic_k", first_dyn.get("k", 0))
            row.setdefault("hit_at_dynamic_k", first_dyn.get("hit", 0))
            row.setdefault("dynamic_by_ratio", json.dumps(dyn_map, ensure_ascii=False))
            for ratio in dynamic_topk_ratios:
                ratio_key = f"{ratio:.4f}"
                ratio_slug = f"{ratio:.2f}".replace(".", "_")
                dyn = dyn_map.get(ratio_key, {})
                row.setdefault(f"dynamic_k_r{ratio_slug}", dyn.get("k", 0))
                row.setdefault(f"hit_at_dynamic_k_r{ratio_slug}", dyn.get("hit", 0))
            writer.writerow(row)


def write_retrieval_eval_md(
    run_dir: Path,
    runs: list[dict],
    eval_topks: list[int],
    dynamic_topk_ratios: list[float],
) -> None:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Metric: strict chunk-id matching against the fixed reference chunks file.",
        f"- Dynamic top-k ratios: [{', '.join(f'{r:.2f}' for r in dynamic_topk_ratios)}]",
        "",
        "## Per-Run Metrics",
        "",
    ]
    dyn_headers = [f"Hit@DynamicK(r={ratio:.2f}) | AvgK(r={ratio:.2f})" for ratio in dynamic_topk_ratios]
    header = (
        "| Tag | Questions | "
        + " | ".join([f"Hit@{k} (count/rate)" for k in eval_topks] + dyn_headers)
        + " |"
    )
    sep = "| --- | ---: | " + " | ".join(["---:"] * (len(eval_topks) + (2 * len(dynamic_topk_ratios)))) + " |"
    lines.append(header)
    lines.append(sep)
    for run in runs:
        evaluated = run.get("retrieval_eval_total", len(run.get("retrieval_eval", [])))
        hit_counts = run.get("retrieval_hit_counts", {})
        hit_rates = run.get("retrieval_hit_rates", {})
        metric_cells = []
        for k in eval_topks:
            count = hit_counts.get(str(k), 0)
            rate = hit_rates.get(str(k))
            rate_txt = f"{rate:.3f}" if rate is not None else "n/a"
            metric_cells.append(f"{count}/{evaluated} ({rate_txt})")
        dyn_by_ratio = run.get("retrieval_dynamic_by_ratio", {})
        dyn_cells = []
        for ratio in dynamic_topk_ratios:
            key = f"{ratio:.4f}"
            stat = dyn_by_ratio.get(key, {})
            dyn_count = stat.get("hit_count", 0)
            dyn_rate = stat.get("hit_rate")
            dyn_k_avg = stat.get("avg_k")
            dyn_rate_txt = f"{dyn_rate:.3f}" if dyn_rate is not None else "n/a"
            dyn_k_avg_txt = f"{dyn_k_avg:.2f}" if dyn_k_avg is not None else "n/a"
            dyn_cells.extend([f"{dyn_count}/{evaluated} ({dyn_rate_txt})", dyn_k_avg_txt])
        lines.append(
            f"| {run['tag']} | {evaluated} | "
            + " | ".join(metric_cells)
            + " | "
            + " | ".join(dyn_cells)
            + " |"
        )

    lines.extend(["", "## Per-Question Top1 View", ""])
    for run in runs:
        lines.append(f"### {run['tag']}")
        lines.append("")
        top1_key = f"hit_at_{eval_topks[0]}"
        lines.append("| Q | Top1 | DynamicK/Hit By Ratio | Top1 Score | Top1 Matched Chunk IDs | Top1 Preview |")
        lines.append("| ---: | :---: | --- | ---: | --- | --- |")
        for item in run.get("retrieval_eval", []):
            score = item.get("top1_score")
            dyn_map = item.get("dynamic_by_ratio", {})
            dyn_txt_parts = []
            for ratio in dynamic_topk_ratios:
                key = f"{ratio:.4f}"
                dyn = dyn_map.get(key, {})
                dyn_txt_parts.append(f"r={ratio:.2f}:K={dyn.get('k', 0)},{'✅' if dyn.get('hit', 0) == 1 else '❌'}")
            dyn_txt = "<br>".join(dyn_txt_parts)
            lines.append(
                f"| {item.get('q_id', '')} | "
                f"{'✅' if item.get(top1_key) == 1 else '❌'} | "
                f"{dyn_txt} | "
                f"{(f'{score:.4f}' if score is not None else '')} | "
                f"{item.get('top1_matched_chunk_ids', '')} | "
                f"{item.get('top1_preview', '')} |"
            )
        lines.append("")

    (run_dir / "retrieval_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                lines.append("#### Prompt sent to LLM")
                lines.append("")
                lines.append(prompt_text)
                lines.append("")

            # Retrieval details block
            block = item.get("block", {})
            retrieval_detail = block.get("cleaned_text", "").strip()
            if not retrieval_detail:
                event = item.get("event", {})
                retrieval_detail = render_retrieval_fallback(event) if event else "_No retrieval trace captured._"

            lines.append("#### Retrieval details")
            lines.append("")
            lines.append(retrieval_detail)
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retrieval_full_md(run_dir: Path, runs: list[dict]) -> None:
    output_path = run_dir / "retrieval_full.md"
    lines = [
        "# Retrieval Full Results",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Purpose: full base retrieval and rerank details per question/doc",
        "",
    ]

    for run in runs:
        lines.append(f"## {run['tag']}")
        lines.append("")

        answers = run.get("answers", [])
        retrieval_blocks = run.get("retrieval_md_blocks", [])
        retrieval_events = run.get("retrieval_events", [])
        parsed_blocks = [parse_retrieval_markdown_block(block) for block in retrieval_blocks]
        item_count = max(len(answers), len(parsed_blocks), len(retrieval_events))

        if item_count == 0:
            lines.append("_No retrieval data captured for this run._")
            lines.append("")
            continue

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

            lines.append(f"### Q{q_id}")
            lines.append("")
            lines.append(f"- Question: `{question}`")

            base_candidates = block_item.get("base_candidates")
            reranked_kept = block_item.get("reranked_kept")
            if base_candidates is None:
                base_candidates = event_item.get("base_count", "-")
            if reranked_kept is None:
                reranked_kept = event_item.get("rerank_count", "-")
            lines.append(f"- Base candidates: {base_candidates}")
            lines.append(f"- Reranked kept: {reranked_kept}")
            lines.append("")

            retrieval_detail = block_item.get("cleaned_text", "").strip()
            if not retrieval_detail:
                retrieval_detail = (
                    render_retrieval_fallback(event_item, max_items=None)
                    if event_item
                    else "_No retrieval trace captured._"
                )

            lines.append(retrieval_detail)
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    question_file = resolve_path(project_root, args.question_file).resolve()
    gold_file = resolve_path(project_root, args.gold_file).resolve()
    gold_chunks_file = resolve_path(project_root, args.gold_chunks_file).resolve() if args.gold_chunks_file else None
    eval_topks = parse_eval_topks(args.eval_topk)
    dynamic_topk_ratios = parse_dynamic_topk_ratios(args.dynamic_topk_ratio)
    max_eval_k = max(eval_topks)
    docs = [resolve_path(project_root, d).resolve() for d in args.docs]
    output_base = resolve_path(project_root, args.output_dir).resolve()
    ollama_info = ensure_ollama(args)

    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    gold_map = load_gold_map(gold_file if gold_file.exists() else None)
    chunk_catalog = {}
    chunk_catalog_source = ""
    if gold_chunks_file and gold_chunks_file.exists():
        chunk_catalog = load_chunk_catalog(gold_chunks_file)
        chunk_catalog_source = str(gold_chunks_file)
    if not docs:
        raise ValueError("No doc files provided.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"format_compare_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    default_gold_chunks_file = answers_dir / "master_md.chunks.jsonl"
    print(f"[INFO] Project root: {project_root}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
    print(f"[INFO] Gold file: {gold_file if gold_file.exists() else '[missing]'}", flush=True)
    if gold_chunks_file:
        print(f"[INFO] Gold chunks file: {gold_chunks_file if gold_chunks_file.exists() else '[missing]'}", flush=True)
    else:
        print(f"[INFO] Gold chunks file: [auto] {default_gold_chunks_file}", flush=True)
    print(f"[INFO] Gold entries loaded: {len(gold_map)}", flush=True)
    print(f"[INFO] Gold chunks loaded: {len(chunk_catalog.get('chunks', []))}", flush=True)
    print(f"[INFO] Retrieval eval top-k: {eval_topks}", flush=True)
    print(f"[INFO] Retrieval dynamic top-k ratios: {[round(r, 2) for r in dynamic_topk_ratios]}", flush=True)
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
                "retrieval_eval": [],
                "retrieval_eval_total": 0,
                "retrieval_hit_counts": {},
                "retrieval_hit_rates": {},
                "retrieval_dynamic_hit_count": 0,
                "retrieval_dynamic_hit_rate": None,
                "retrieval_dynamic_k_avg": None,
                "retrieval_hit_at_1": None,
                "retrieval_hit_at_5": None,
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

        # Auto-bind strict evaluation chunks file to current run's master_md output if not explicitly provided.
        if not chunk_catalog:
            if gold_chunks_file and gold_chunks_file.exists():
                chunk_catalog = load_chunk_catalog(gold_chunks_file)
                chunk_catalog_source = str(gold_chunks_file)
            elif tag == "master_md" and chunk_path.exists():
                chunk_catalog = load_chunk_catalog(chunk_path)
                chunk_catalog_source = str(chunk_path)
            elif default_gold_chunks_file.exists():
                chunk_catalog = load_chunk_catalog(default_gold_chunks_file)
                chunk_catalog_source = str(default_gold_chunks_file)

            if chunk_catalog:
                print(
                    f"[RUN {idx}/{len(docs)}] Loaded strict eval chunk catalog from: {chunk_catalog_source} "
                    f"(chunks={len(chunk_catalog.get('chunks', []))})",
                    flush=True,
                )

        answers = parse_answers(answer_path)
        retrieval_md_blocks = extract_retrieval_markdown_blocks(log_text)
        parsed_rerank_blocks = [parse_rerank_rows_from_block(block) for block in retrieval_md_blocks]
        retrieval_eval_rows = []
        for q_idx, answer_item in enumerate(answers):
            question = answer_item.get("question", "")
            gold = gold_map.get(question, {})
            gold_chunk = str(gold.get("gold_chunk", ""))
            gold_chunk_ids = list(gold.get("gold_chunk_ids", []))
            expected_answer = str(gold.get("expected_answer", ""))
            rerank_rows = parsed_rerank_blocks[q_idx]["rerank_rows"] if q_idx < len(parsed_rerank_blocks) else []
            rerank_rows = rerank_rows[:max_eval_k]
            hit_map = {k: 0 for k in eval_topks}
            top1_matched_chunk_ids = []
            dynamic_by_ratio = {f"{ratio:.4f}": {"k": 0, "hit": 0} for ratio in dynamic_topk_ratios}

            if rerank_rows and gold_chunk_ids and chunk_catalog.get("chunks"):
                for row in rerank_rows:
                    row["matched_chunk_ids"] = match_preview_to_chunk_ids(row.get("preview", ""), chunk_catalog)
                top1_matched_chunk_ids = rerank_rows[0].get("matched_chunk_ids", [])
                for k in eval_topks:
                    considered = rerank_rows[:k]
                    hit_map[k] = 1 if any(
                        any(cid in gold_chunk_ids for cid in row.get("matched_chunk_ids", []))
                        for row in considered
                    ) else 0
                max_score = rerank_rows[0].get("score")
                if isinstance(max_score, (int, float)):
                    max_score_f = float(max_score)
                    for ratio in dynamic_topk_ratios:
                        threshold = max_score_f * ratio
                        dynamic_rows = [
                            row for row in rerank_rows
                            if isinstance(row.get("score"), (int, float)) and float(row.get("score")) >= threshold
                        ]
                        dynamic_k = len(dynamic_rows)
                        dynamic_hit = 1 if any(
                            any(cid in gold_chunk_ids for cid in row.get("matched_chunk_ids", []))
                            for row in dynamic_rows
                        ) else 0
                        dynamic_by_ratio[f"{ratio:.4f}"] = {"k": dynamic_k, "hit": dynamic_hit}

            top1 = rerank_rows[0] if rerank_rows else None
            first_ratio = dynamic_topk_ratios[0]
            first_ratio_key = f"{first_ratio:.4f}"
            first_dynamic = dynamic_by_ratio.get(first_ratio_key, {"k": 0, "hit": 0})
            row = {
                "q_id": answer_item.get("q_id", q_idx + 1),
                "question": question,
                "gold_chunk_ids": "|".join(str(cid) for cid in gold_chunk_ids),
                "gold_chunk": gold_chunk,
                "expected_answer": expected_answer,
                "top1_matched_chunk_ids": "|".join(str(cid) for cid in top1_matched_chunk_ids),
                "top1_score": top1.get("score") if top1 else None,
                "top1_preview": top1.get("preview", "") if top1 else "",
                "dynamic_k": first_dynamic.get("k", 0),
                "dynamic_ratio": first_ratio_key,
                "hit_at_dynamic_k": first_dynamic.get("hit", 0),
                "dynamic_by_ratio": dynamic_by_ratio,
            }
            for k in eval_topks:
                row[f"hit_at_{k}"] = hit_map.get(k, 0)
            retrieval_eval_rows.append(row)

        retrieval_hit_counts = {str(k): 0 for k in eval_topks}
        retrieval_hit_rates = {str(k): None for k in eval_topks}
        retrieval_dynamic_hit_count = 0
        retrieval_dynamic_hit_rate = None
        retrieval_dynamic_k_avg = None
        retrieval_dynamic_by_ratio = {
            f"{ratio:.4f}": {"hit_count": 0, "hit_rate": None, "avg_k": None}
            for ratio in dynamic_topk_ratios
        }
        if retrieval_eval_rows:
            total_eval = len(retrieval_eval_rows)
            for k in eval_topks:
                key = f"hit_at_{k}"
                hit_count = sum(int(item.get(key, 0)) for item in retrieval_eval_rows)
                retrieval_hit_counts[str(k)] = hit_count
                retrieval_hit_rates[str(k)] = hit_count / total_eval
            for ratio in dynamic_topk_ratios:
                ratio_key = f"{ratio:.4f}"
                hit_count = sum(
                    int(item.get("dynamic_by_ratio", {}).get(ratio_key, {}).get("hit", 0))
                    for item in retrieval_eval_rows
                )
                avg_k = sum(
                    int(item.get("dynamic_by_ratio", {}).get(ratio_key, {}).get("k", 0))
                    for item in retrieval_eval_rows
                ) / total_eval
                retrieval_dynamic_by_ratio[ratio_key] = {
                    "hit_count": hit_count,
                    "hit_rate": hit_count / total_eval,
                    "avg_k": avg_k,
                }
            first_ratio_key = f"{dynamic_topk_ratios[0]:.4f}"
            first_stat = retrieval_dynamic_by_ratio.get(first_ratio_key, {})
            retrieval_dynamic_hit_count = int(first_stat.get("hit_count", 0))
            retrieval_dynamic_hit_rate = first_stat.get("hit_rate")
            retrieval_dynamic_k_avg = first_stat.get("avg_k")

        retrieval_hit_at_1 = retrieval_hit_rates.get("1")
        retrieval_hit_at_5 = retrieval_hit_rates.get("5")

        runs.append({
            "tag": tag,
            "exit_code": proc.returncode,
            "duration_sec": duration_sec,
            "doc_path": str(doc),
            "log_path": str(log_path),
            "answer_path": str(answer_path),
            "chunk_path": str(chunk_path),
            "loaded_line": extract_metric_line(log_text, r"^\[DEBUG\] Loaded documents:.*$"),
            "chunks_line": extract_metric_line(log_text, r"^\[DEBUG\] Total chunks after split:.*$"),
            "embedding_line": extract_metric_line(log_text, r"^\[DEBUG\] Embedding \+ vector DB build time:.*$"),
            "prepare_line": extract_metric_line(log_text, r"^\[DEBUG\] prepare_data total time:.*$"),
            "qa_time_lines": extract_all_metric_lines(log_text, r"^\[DEBUG\] QA invoke time:.*$"),
            "qa_times_sec": extract_qa_times(log_text),
            "retrieval_calls": len(extract_all_metric_lines(log_text, r"^\[DEBUG\]\[RETRIEVE\] q=.*$")),
            "retrieval_base_stats": extract_all_metric_lines(log_text, r"^\[DEBUG\]\[RETRIEVE\]\[BASE_STATS\] .*$"),
            "retrieval_rerank_stats": extract_all_metric_lines(log_text, r"^\[DEBUG\]\[RETRIEVE\]\[RERANK_STATS\] .*$"),
            "retrieval_base_top": extract_all_metric_lines(log_text, r"^\[DEBUG\]\[RETRIEVE\]\[BASE_TOP\] .*$"),
            "retrieval_rerank_top": extract_all_metric_lines(log_text, r"^\[DEBUG\]\[RETRIEVE\]\[RERANK_TOP\] .*$"),
            "retrieval_md_blocks": retrieval_md_blocks,
            "prompt_md_blocks": extract_prompt_markdown_blocks(log_text),
            "retrieval_events": extract_retrieval_events(log_text),
            "answers": answers,
            "chunk_summary": chunk_summary,
            "error_hint": error_hint,
            "retrieval_eval": retrieval_eval_rows,
            "retrieval_eval_total": len(retrieval_eval_rows),
            "retrieval_hit_counts": retrieval_hit_counts,
            "retrieval_hit_rates": retrieval_hit_rates,
            "retrieval_dynamic_by_ratio": retrieval_dynamic_by_ratio,
            "retrieval_dynamic_hit_count": retrieval_dynamic_hit_count,
            "retrieval_dynamic_hit_rate": retrieval_dynamic_hit_rate,
            "retrieval_dynamic_k_avg": retrieval_dynamic_k_avg,
            "retrieval_hit_at_1": retrieval_hit_at_1,
            "retrieval_hit_at_5": retrieval_hit_at_5,
        })

    write_summary_md(run_dir, question_file, runs, eval_topks, dynamic_topk_ratios)
    write_comparison_csv(run_dir, runs)
    write_retrieval_trace_md(run_dir, runs)
    write_retrieval_full_md(run_dir, runs)
    write_retrieval_eval_csv(run_dir, runs, eval_topks, dynamic_topk_ratios)
    write_retrieval_eval_md(run_dir, runs, eval_topks, dynamic_topk_ratios)
    (run_dir / "summary.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Experiment done.", flush=True)
    print(f"[INFO] Output directory: {run_dir}", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Comparison CSV: {run_dir / 'comparison.csv'}", flush=True)
    print(f"[INFO] Retrieval Trace: {run_dir / 'retrieval_trace.md'}", flush=True)
    print(f"[INFO] Retrieval Full: {run_dir / 'retrieval_full.md'}", flush=True)
    print(f"[INFO] Retrieval Eval CSV: {run_dir / 'retrieval_eval.csv'}", flush=True)
    print(f"[INFO] Retrieval Eval MD: {run_dir / 'retrieval_eval.md'}", flush=True)


if __name__ == "__main__":
    main()
