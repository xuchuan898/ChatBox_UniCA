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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        "--gold-file",
        type=Path,
        default=None,
        help=(
            "Optional gold file for layered retrieval scoring. "
            "If provided, the script computes gold-hit metrics from rerank top-k rows."
        ),
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
        "--layer1-k",
        type=int,
        default=5,
        help="First evaluation layer size for layered scoring (default: 5).",
    )
    parser.add_argument(
        "--layer2-k",
        type=int,
        default=8,
        help="Second evaluation layer size for layered scoring (default: 8).",
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


def configured_model_name(config_path: Path, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    try:
        from core.config_loader import load_config
        config = load_config(config_path)
        return str(config["generation"]["model_name"])
    except Exception:
        return "qwen2.5:32b"


def ensure_ollama(args: argparse.Namespace, model_name: str) -> dict:
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
            if len(cells) >= 7:
                chunk_id_cell = cells[1]
                score_cell = cells[2]
                source_cell = cells[3]
                chunk_type_cell = cells[5]
                preview_cell = cells[6]
            else:
                chunk_id_cell = ""
                score_cell = cells[1]
                source_cell = cells[2]
                chunk_type_cell = cells[4]
                preview_cell = cells[5]
            rerank_rows.append(
                {
                    "rank": int(cells[0]),
                    "chunk_id": int(chunk_id_cell) if chunk_id_cell not in {"", "-", "unknown"} else None,
                    "score": float(score_cell),
                    "source": source_cell,
                    "chunk_type": chunk_type_cell,
                    "preview": preview_cell,
                }
            )

    return {"question": question, "rerank_rows": rerank_rows[:top_k]}


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", (text or "").casefold())).strip()


def load_gold_index(gold_file: Path) -> dict[str, list[dict]]:
    raw = json.loads(gold_file.read_text(encoding="utf-8"))
    items = raw.get("items", []) if isinstance(raw, dict) else raw
    index: dict[str, list[dict]] = {}
    for item in items:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        index.setdefault(_normalize_text(question), []).append(item)
    return index


def _gold_ids_from_item(item: dict) -> list[int]:
    raw_ids = item.get("gold_chunk_ids") or item.get("gold_chunks_ids") or []
    ids: list[int] = []
    for value in raw_ids:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def _gold_texts_from_item(item: dict) -> list[str]:
    texts: list[str] = []
    gold_chunk = item.get("gold_chunk")
    if gold_chunk:
        texts.append(str(gold_chunk))
    gold_chunks_full = item.get("gold_chunks_full")
    if isinstance(gold_chunks_full, list):
        texts.extend(str(v) for v in gold_chunks_full if v)
    return [t for t in texts if t.strip()]


def _row_matches_gold_text(row: dict, gold_texts: list[str]) -> bool:
    if not gold_texts:
        return False
    row_text = _normalize_text(f"{row.get('source', '')} {row.get('chunk_type', '')} {row.get('preview', '')}")
    for gold_text in gold_texts:
        gold_norm = _normalize_text(gold_text)
        if not gold_norm:
            continue
        if gold_norm in row_text or row_text in gold_norm:
            return True
    return False


def score_layered_hits(rerank_rows: list[dict], gold_item: dict | None, layer1_k: int, layer2_k: int) -> dict:
    if not gold_item:
        return {
            "gold_total": None,
            "gold_hits_top_layer1": None,
            "gold_hits_layer2": None,
            "gold_hits_total": None,
            "gold_layered_score": None,
            "gold_full_recall": None,
            "gold_layer1_recall": None,
            "gold_layer2_recall": None,
            "gold_hit_ids": "",
        }

    gold_ids = _gold_ids_from_item(gold_item)
    gold_texts = _gold_texts_from_item(gold_item)
    gold_total = len(gold_ids) if gold_ids else len(gold_texts)
    if gold_total <= 0:
        return {
            "gold_total": 0,
            "gold_hits_top_layer1": 0,
            "gold_hits_layer2": 0,
            "gold_hits_total": 0,
            "gold_layered_score": 0.0,
            "gold_full_recall": 0.0,
            "gold_layer1_recall": 0.0,
            "gold_layer2_recall": 0.0,
            "gold_hit_ids": "",
        }

    hit_ids: list[str] = []
    top_layer1_hits: set[int] = set()
    layer2_hits: set[int] = set()
    total_hits: set[int] = set()
    text_hits_top_layer1 = 0
    text_hits_layer2 = 0
    text_hits_total = 0

    for row in rerank_rows:
        rank = int(row.get("rank") or 0)
        chunk_id = row.get("chunk_id")
        matched = False

        if gold_ids and chunk_id is not None and int(chunk_id) in gold_ids:
            matched = True
            total_hits.add(int(chunk_id))
            hit_ids.append(str(int(chunk_id)))
            if rank <= layer1_k:
                top_layer1_hits.add(int(chunk_id))
            if layer1_k < rank <= layer2_k:
                layer2_hits.add(int(chunk_id))
        elif not gold_ids and _row_matches_gold_text(row, gold_texts):
            matched = True
            if rank <= layer1_k:
                text_hits_top_layer1 += 1
            if layer1_k < rank <= layer2_k:
                text_hits_layer2 += 1
            text_hits_total += 1

        if matched and gold_ids:
            continue

    if gold_ids:
        gold_hits_top_layer1 = len(top_layer1_hits)
        gold_hits_layer2 = len(layer2_hits)
        gold_hits_total = len(total_hits)
    else:
        gold_hits_top_layer1 = text_hits_top_layer1
        gold_hits_layer2 = text_hits_layer2
        gold_hits_total = text_hits_total

    return {
        "gold_total": gold_total,
        "gold_hits_top_layer1": gold_hits_top_layer1,
        "gold_hits_layer2": gold_hits_layer2,
        "gold_hits_total": gold_hits_total,
        "gold_layered_score": round(gold_hits_total / gold_total, 4),
        "gold_full_recall": round(gold_hits_total / gold_total, 4),
        "gold_layer1_recall": round(gold_hits_top_layer1 / gold_total, 4),
        "gold_layer2_recall": round(gold_hits_layer2 / gold_total, 4),
        "gold_hit_ids": ",".join(hit_ids),
    }


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


def build_summary_md(
    run_dir: Path,
    summary_rows: list[dict],
    baseline_alpha: float,
    top_k: int,
    gold_enabled: bool = False,
    layer1_k: int = 5,
    layer2_k: int = 8,
) -> None:
    lines = [
        "# Rerank Alpha Experiment Summary",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Fixed retrieval weights: `weight_vec=1.25`, `weight_bm25=0.75`, `secondary_variant_weight=0.85`",
        f"- Variant mode: `mapped_current`",
        f"- Rerank details shown per question: Top-{top_k}",
        f"- Baseline alpha: `{baseline_alpha:.2f}`",
    ]
    if gold_enabled:
        lines.extend(
            [
                f"- Gold scoring: layer1=`top-{layer1_k}`, layer2=`top-{layer2_k}`, score=`hits_total / gold_total`",
                f"- 金标准评分 / Gold scoring (EN): first check top-{layer1_k}, then use ranks {layer1_k + 1}-{layer2_k} to complete remaining golds.",
            ]
        )
    lines.extend([
        "",
        "| Alpha | Questions | Avg Top1 Score | Avg Margin(1-2) | Unique Top1 Chunks | Top1 Changed vs Baseline |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['questions']} | {row['avg_rerank_top1_score']:.4f} | "
            f"{row['avg_rerank_margin_top1_top2']:.4f} | {row['unique_top1_chunks']} | "
            f"{row['top1_changed_rate_vs_baseline']:.4f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_details_md(
    run_dir: Path,
    questions: list[str],
    run_payloads: list[dict],
    top_k: int,
    gold_enabled: bool = False,
) -> None:
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
            if gold_enabled:
                gold = run.get("gold_rows", [])
                gold_row = gold[q_idx - 1] if q_idx - 1 < len(gold) else None
                if gold_row:
                    lines.append(
                        f"- Gold score: `{gold_row.get('gold_layered_score', 0.0):.4f}` "
                        f"({gold_row.get('gold_hits_total', 0)}/{gold_row.get('gold_total', 0)})"
                    )
                    hit_ids = str(gold_row.get('gold_hit_ids', '')) or '-'
                    lines.append(
                        f"- Gold layer1 hits: `{gold_row.get('gold_hits_top_layer1', 0)}` | "
                        f"layer2 hits: `{gold_row.get('gold_hits_layer2', 0)}` | "
                        f"hit ids: `{_md_escape(hit_ids)}`"
                    )
                    lines.append("")
            lines.append(f"Rerank Top-{top_k}:")
            lines.append("| Rank | Chunk ID | Score | Source | Chunk Type | Preview |")
            lines.append("| ---: | ---: | ---: | --- | --- | --- |")
            if block.get("rerank_rows"):
                for row in block["rerank_rows"]:
                    lines.append(
                        f"| {row['rank']} | {row.get('chunk_id', '')} | {row['score']:.4f} | {_md_escape(row['source'])} | "
                        f"{_md_escape(row['chunk_type'])} | {_md_escape(row['preview'])} |"
                    )
            else:
                lines.append("| - | - | - | - | - | _No rerank rows captured_ |")
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
    if args.layer1_k <= 0 or args.layer2_k <= 0:
        raise ValueError("--layer1-k and --layer2-k must be > 0")
    if args.layer1_k > args.layer2_k:
        raise ValueError("--layer1-k must be <= --layer2-k")

    alphas = sorted(set(args.alphas))
    if not alphas:
        raise ValueError("--alphas must not be empty")
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")

    questions = load_questions(question_file)
    if not questions:
        raise ValueError(f"No valid questions in {question_file}")

    gold_index = {}
    gold_file = None
    if args.gold_file is not None:
        gold_file = resolve_path(project_root, args.gold_file).resolve()
        if not gold_file.exists():
            raise FileNotFoundError(f"Gold file not found: {gold_file}")
        gold_index = load_gold_index(gold_file)

    model_name = configured_model_name(Path(args.project_root) / "config.yaml" if hasattr(args, 'project_root') else project_root / "config.yaml", None)
    ollama_info = ensure_ollama(args, model_name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"rerank_alpha_experiment_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Doc file: {doc_file}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
    if gold_file:
        print(f"[INFO] Gold file: {gold_file}", flush=True)
        print(f"[INFO] Gold layers: top-{args.layer1_k} + top-{args.layer2_k}", flush=True)
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
        parse_top_k = max(args.top_k, args.layer2_k)
        retrieval_blocks = [parse_retrieval_block(b, parse_top_k) for b in extract_retrieval_markdown_blocks(log_text)]
        variant_events = extract_query_variant_events(log_text)
        dual_events = extract_dual_query_events(log_text)

        rows_for_alpha = []
        gold_rows_for_alpha = []
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

            gold_item = None
            gold_key = _normalize_text(question)
            candidates = gold_index.get(gold_key, [])
            if candidates:
                if len(candidates) == 1:
                    gold_item = candidates[0]
                else:
                    lang_candidates = [item for item in candidates if str(item.get("lang", "")).strip().lower() == lang]
                    gold_item = lang_candidates[0] if lang_candidates else candidates[0]

            gold_metrics = score_layered_hits(rerank_rows, gold_item, args.layer1_k, args.layer2_k)

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
                **gold_metrics,
            }
            rows_for_alpha.append(row)
            question_level_rows.append(row.copy())
            gold_rows_for_alpha.append(gold_metrics)

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
        if gold_index:
            summary["avg_gold_total"] = round(
                statistics.mean([r["gold_total"] for r in gold_rows_for_alpha if r["gold_total"] is not None]), 4
            ) if any(r["gold_total"] is not None for r in gold_rows_for_alpha) else 0.0
            summary["avg_gold_hits_total"] = round(
                statistics.mean([r["gold_hits_total"] for r in gold_rows_for_alpha if r["gold_hits_total"] is not None]), 4
            ) if any(r["gold_hits_total"] is not None for r in gold_rows_for_alpha) else 0.0
            summary["avg_gold_layered_score"] = round(
                statistics.mean([r["gold_layered_score"] for r in gold_rows_for_alpha if r["gold_layered_score"] is not None]), 4
            ) if any(r["gold_layered_score"] is not None for r in gold_rows_for_alpha) else 0.0
            summary["avg_gold_layer1_recall"] = round(
                statistics.mean([r["gold_layer1_recall"] for r in gold_rows_for_alpha if r["gold_layer1_recall"] is not None]), 4
            ) if any(r["gold_layer1_recall"] is not None for r in gold_rows_for_alpha) else 0.0
            summary["avg_gold_layer2_recall"] = round(
                statistics.mean([r["gold_layer2_recall"] for r in gold_rows_for_alpha if r["gold_layer2_recall"] is not None]), 4
            ) if any(r["gold_layer2_recall"] is not None for r in gold_rows_for_alpha) else 0.0
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
                "gold_rows": gold_rows_for_alpha,
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
            "avg_gold_total",
            "avg_gold_hits_total",
            "avg_gold_layered_score",
            "avg_gold_layer1_recall",
            "avg_gold_layer2_recall",
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
            "gold_total",
            "gold_hits_top_layer1",
            "gold_hits_layer2",
            "gold_hits_total",
            "gold_layered_score",
            "gold_full_recall",
            "gold_layer1_recall",
            "gold_layer2_recall",
            "gold_hit_ids",
        ],
    )

    (run_dir / "runs.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc_file": str(doc_file),
                "question_file": str(question_file),
                "top_k": args.top_k,
                "gold_file": str(gold_file) if gold_file else None,
                "layer1_k": args.layer1_k,
                "layer2_k": args.layer2_k,
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

    build_summary_md(run_dir, summary_rows, baseline_alpha, args.top_k, bool(gold_index), args.layer1_k, args.layer2_k)
    build_details_md(run_dir, questions, run_payloads, args.top_k, bool(gold_index))

    print("[INFO] Alpha experiment finished.", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Details: {run_dir / 'details.md'}", flush=True)
    print(f"[INFO] Alpha summary CSV: {run_dir / 'alpha_summary.csv'}", flush=True)
    print(f"[INFO] Question-level CSV: {run_dir / 'alpha_question_level.csv'}", flush=True)


if __name__ == "__main__":
    main()
