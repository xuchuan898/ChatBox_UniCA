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

DEFAULT_SWEEP_VALUES = [round(x * 0.1, 1) for x in range(11)]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run focused ablation for query expansion ratio with fixed bm25 ratio, "
            "fixed rerank alpha, and rerank-candidates sweep."
        )
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
        help="Single document used in the ablation.",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("questions/questions_batch_student_short_typo_en_fr.txt"),
        help="Question file path.",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("questions/questions_batch_student_short_typo_en_fr_gold.json"),
        help="Gold file with expected answers and strict gold chunk ids.",
    )
    parser.add_argument(
        "--gold-chunks-file",
        type=Path,
        default=None,
        help=(
            "Reference chunks jsonl for strict retrieval evaluation by chunk id. "
            "If omitted, uses current run output: answers/master_md.chunks.jsonl."
        ),
    )
    parser.add_argument(
        "--bm25-ratios",
        type=float,
        nargs="*",
        default=[0.2],
        help="Fixed BM25 ratio list (default: 0.2).",
    )
    parser.add_argument(
        "--translation-ratios",
        type=float,
        nargs="*",
        default=DEFAULT_SWEEP_VALUES,
        help="Query expansion ratio values in [0,1] (mapped to secondary_variant_weight sweep).",
    )
    parser.add_argument(
        "--rerank-alphas",
        type=float,
        nargs="*",
        default=[0.5],
        help="Fixed rerank alpha list (default: 0.5).",
    )
    parser.add_argument(
        "--fixed-bm25-ratio",
        type=float,
        default=0.2,
        help="Fixed bm25 ratio used when sweeping other variables.",
    )
    parser.add_argument(
        "--fixed-translation-ratio",
        type=float,
        default=0.5,
        help="Fixed retrieval translation ratio used when sweeping other variables.",
    )
    parser.add_argument(
        "--fixed-rerank-alpha",
        type=float,
        default=0.5,
        help="Fixed rerank alpha used when sweeping other variables.",
    )
    parser.add_argument(
        "--variant-mode",
        type=str,
        default="mapped_current",
        choices=["primary_only", "mapped_current", "mapped_expanded"],
        help="Variant generation mode.",
    )
    parser.add_argument(
        "--eval-topk",
        type=str,
        default="1,5,8",
        help="Comma-separated top-k values for strict retrieval evaluation.",
    )
    parser.add_argument(
        "--dynamic-topk-ratio",
        type=str,
        default="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
        help=(
            "Comma-separated dynamic top-k ratios. "
            "For each ratio r, keep candidates with score >= r * max_score."
        ),
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        nargs="*",
        default=[10, 20, 30],
        help="List of candidate counts to send to rerank stage (e.g. --rerank-candidates 10 20 30 40).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k rows shown in details markdown (default: 5).",
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
    parser.add_argument("--enable-query-expansion", type=str, default=None)
    parser.add_argument("--expansion-model", type=str, default=None)
    parser.add_argument("--expansion-paraphrases", type=int, default=None)
    parser.add_argument("--expansion-add-translation", type=str, default=None)
    parser.add_argument("--expansion-source-lang", type=str, default=None)
    parser.add_argument("--enable-multi-variant-rerank", type=str, default="false")
    parser.add_argument("--enable-cache", type=str, default="false")
    parser.add_argument("--enable-memory", type=str, default="false")
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


def _validate_ratio_list(name: str, values: list[float]) -> list[float]:
    if not values:
        raise ValueError(f"--{name} must not be empty.")
    cleaned = sorted(set(round(float(v), 4) for v in values))
    for v in cleaned:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"--{name} values must be in [0,1], got: {v}")
    return cleaned


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
    questions: list[str] = []
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


def _weight_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_run_plan(args: argparse.Namespace) -> list[dict]:
    bm25_ratios = _validate_ratio_list("bm25-ratios", args.bm25_ratios)
    translation_ratios = _validate_ratio_list("translation-ratios", args.translation_ratios)
    rerank_alphas = _validate_ratio_list("rerank-alphas", args.rerank_alphas)

    # integer list for how many candidates to send into rerank
    rerank_candidates_raw = args.rerank_candidates or [30]
    rerank_candidates = sorted(set(int(x) for x in rerank_candidates_raw))

    fixed_bm25_ratio = round(float(args.fixed_bm25_ratio), 4)
    fixed_translation_ratio = round(float(args.fixed_translation_ratio), 4)
    fixed_rerank_alpha = round(float(args.fixed_rerank_alpha), 4)
    for name, v in (
        ("fixed-bm25-ratio", fixed_bm25_ratio),
        ("fixed-translation-ratio", fixed_translation_ratio),
        ("fixed-rerank-alpha", fixed_rerank_alpha),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"--{name} must be in [0,1], got: {v}")

    bm25_ratio = bm25_ratios[0] if bm25_ratios else fixed_bm25_ratio
    alpha = rerank_alphas[0] if rerank_alphas else fixed_rerank_alpha

    plan = []
    for tr in translation_ratios:
        svw = 1.0 if args.variant_mode == "primary_only" else tr
        for rc in rerank_candidates:
            run_id = (
                f"qe_ratio__r{_weight_slug(tr)}"
                f"__bm25{_weight_slug(bm25_ratio)}__ra{_weight_slug(alpha)}__rc{rc}"
            )
            plan.append(
                {
                    "run_id": run_id,
                    "sweep_group": "query_expansion_ratio",
                    "sweep_var": "query_expansion_ratio",
                    "sweep_value": float(tr),
                    "weight_vec": float(1.0 - bm25_ratio),
                    "weight_bm25": float(bm25_ratio),
                    "variant_mode": args.variant_mode,
                    "secondary_variant_weight": float(svw),
                    "translation_ratio": float(tr),
                    "rerank_alpha": float(alpha),
                    "rerank_candidates": int(rc),
                }
            )
    return plan


def extract_retrieval_markdown_blocks(log_text: str) -> list[str]:
    pattern = re.compile(
        r"^\[DEBUG\]\[RETRIEVE_MD_BEGIN\]\n(.*?)\n\[DEBUG\]\[RETRIEVE_MD_END\]$",
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(log_text)]


def extract_query_variant_events(log_text: str) -> list[dict]:
    events = []
    current = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        start_match = re.match(
            r"^\[DEBUG\]\[RETRIEVE\] q=(['\"])(.*)\1 base_count=\d+ rerank_count=\d+$",
            line,
        )
        if start_match:
            if current:
                events.append(current)
            current = {"question": start_match.group(2), "variants": []}
            continue

        variant_match = re.match(
            r"^\[DEBUG\]\[RETRIEVE\]\[QUERY_VARIANT\] \d+=(['\"])(.*)\1$",
            line,
        )
        if variant_match and current is not None:
            current["variants"].append(variant_match.group(2))

    if current:
        events.append(current)
    return events


def _parse_table_row(row: str) -> list[str]:
    parts = [p.strip() for p in row.strip().split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def parse_retrieval_block(block: str) -> dict:
    question = ""
    section = ""
    base_rows = []
    rerank_rows = []

    def _parse_row(cells: list[str], is_rerank: bool) -> dict | None:
        if len(cells) >= 7:
            return {
                "rank": int(cells[0]),
                "chunk_id": int(cells[1]) if cells[1] not in {"", "-", "unknown"} else None,
                "score": float(cells[2]),
                "source": cells[3],
                "source_type": cells[4],
                "chunk_type": cells[5],
                "preview": cells[6],
            }
        if len(cells) >= 6:
            # Backward compatibility for older logs that omitted Chunk ID.
            return {
                "rank": int(cells[0]),
                "chunk_id": None,
                "score": float(cells[1]),
                "source": cells[2],
                "source_type": cells[3] if not is_rerank else cells[3],
                "chunk_type": cells[4],
                "preview": cells[5],
            }
        return None

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        q_match = re.match(r"^- Question:\s*`(.*)`\s*$", line)
        if q_match:
            question = q_match.group(1).strip()
            continue
        if line.startswith("#### Base Retrieval"):
            section = "base"
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
        if section == "base":
            row = _parse_row(cells, is_rerank=False)
            if row is not None:
                base_rows.append(row)
        elif section == "rerank":
            row = _parse_row(cells, is_rerank=True)
            if row is not None:
                rerank_rows.append(row)

    return {
        "question": question,
        "base_rows": base_rows,
        "rerank_rows": rerank_rows,
    }


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

    matched = [c["chunk_id"] for c in chunks if norm_preview in c["norm_content"]]
    return sorted(set(matched))


def summarize_run_rows(rows: list[dict]) -> dict:
    base_top1_scores = [r["base_top1_score"] for r in rows if r["base_top1_score"] is not None]
    rerank_top1_scores = [r["rerank_top1_score"] for r in rows if r["rerank_top1_score"] is not None]
    rerank_margin = [r["rerank_margin_top1_top2"] for r in rows if r["rerank_margin_top1_top2"] is not None]

    def _safe_mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    return {
        "questions": len(rows),
        "avg_base_top1_score": _safe_mean(base_top1_scores),
        "avg_rerank_top1_score": _safe_mean(rerank_top1_scores),
        "avg_rerank_margin_top1_top2": _safe_mean(rerank_margin),
        "unique_rerank_top1_chunks": len({r["rerank_top1_preview"] for r in rows if r["rerank_top1_preview"]}),
        "avg_hit_at_1": _safe_mean([float(r.get("hit_at_1", 0.0)) for r in rows]),
        "avg_hit_at_5": _safe_mean([float(r.get("hit_at_5", 0.0)) for r in rows]),
        "avg_hit_at_8": _safe_mean([float(r.get("hit_at_8", 0.0)) for r in rows]),
        "avg_hit_at_dynamic_k": _safe_mean([float(r.get("hit_at_dynamic_k", 0.0)) for r in rows]),
        "avg_dynamic_k": _safe_mean([float(r.get("dynamic_k", 0)) for r in rows]) if rows else 0.0,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "'")


def build_markdown_summary(
    run_dir: Path,
    doc_file: Path,
    question_file: Path,
    run_summaries: list[dict],
    question_rows: list[dict],
    top_k: int,
    eval_topks: list[int],
    dynamic_topk_ratios: list[float],
) -> None:
    def _parse_dynamic_map(raw: object) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    lines = [
        "# Retrieval Interaction Ablation Summary",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Doc: {doc_file}",
        f"- Question file: {question_file}",
        f"- Total runs: {len(run_summaries)}",
        f"- Questions per run: {len({r['q_id'] for r in question_rows}) if question_rows else 0}",
        f"- Evaluation top-k: {eval_topks}",
        f"- Dynamic top-k ratios: [{', '.join(f'{r:.2f}' for r in dynamic_topk_ratios)}]",
        "",
        "## Run Configs",
        "",
        "| Run ID | Sweep | Value | Variant Mode | w_vec | w_bm25 | Translation Ratio | Rerank Alpha | Rerank Candidates |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in run_summaries:
        lines.append(
            f"| {row['run_id']} | {row['sweep_group']} | {row['sweep_value']:.2f} | {row['variant_mode']} | "
            f"{row['weight_vec']:.2f} | {row['weight_bm25']:.2f} | {row['translation_ratio']:.2f} | {row['rerank_alpha']:.2f} | {row.get('rerank_candidates', '')} |"
        )

    lines.extend(
        [
            "",
        "## Run Summary",
        "",
        "| Run ID | Sweep | Value | Variant Mode | w_vec | w_bm25 | Translation Ratio | Rerank Alpha | Hit@1 (avg) | Hit@5 (avg) | Hit@8 (avg) | Hit@DynamicK (avg) | Avg Dynamic-K | Avg Rerank Top1 | Avg Margin(1-2) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_summaries:
        h1 = float(row.get("avg_hit_at_1", 0.0))
        h5 = float(row.get("avg_hit_at_5", 0.0))
        h8 = float(row.get("avg_hit_at_8", 0.0))
        hd = float(row.get("avg_hit_at_dynamic_k", 0.0))
        lines.append(
            f"| {row['run_id']} | {row['sweep_group']} | {row['sweep_value']:.2f} | {row['variant_mode']} | "
            f"{row['weight_vec']:.2f} | {row['weight_bm25']:.2f} | {row['translation_ratio']:.2f} | {row['rerank_alpha']:.2f} | {row.get('rerank_candidates', '')} | "
            f"{h1:.4f} | {h5:.4f} | {h8:.4f} | "
            f"{hd:.4f} | {row.get('avg_dynamic_k', 0.0):.2f} | "
            f"{row['avg_rerank_top1_score']:.4f} | {row['avg_rerank_margin_top1_top2']:.4f} |"
        )

    lines.extend(["", "## Dynamic-K Ratio Breakdown", ""])
    dyn_headers = []
    for ratio in dynamic_topk_ratios:
        dyn_headers.extend([f"r={ratio:.2f} Hit(avg)", f"r={ratio:.2f} AvgK"])
    lines.append("| Run ID | " + " | ".join(dyn_headers) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(dyn_headers)) + " |")

    rows_by_run: dict[str, list[dict]] = {}
    for qrow in question_rows:
        run_id = str(qrow.get("run_id", ""))
        rows_by_run.setdefault(run_id, []).append(qrow)

    for row in run_summaries:
        run_id = str(row.get("run_id", ""))
        run_q_rows = rows_by_run.get(run_id, [])
        q = max(len(run_q_rows), 1)
        cells = []
        for ratio in dynamic_topk_ratios:
            key = f"{ratio:.4f}"
            hit_sum = 0.0
            sum_k = 0
            for qrow in run_q_rows:
                dyn_map = _parse_dynamic_map(qrow.get("dynamic_by_ratio", {}))
                dyn = dyn_map.get(key, {})
                if isinstance(dyn, dict):
                    hit_sum += float(dyn.get("hit", 0.0))
                    sum_k += int(dyn.get("k", 0))
            hit_avg = (hit_sum / q) if q else 0.0
            avg_k = (sum_k / q) if q else 0.0
            cells.append(f"{hit_avg:.4f}")
            cells.append(f"{avg_k:.2f}")
        lines.append(f"| {run_id} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            f"- Run Summary table keeps legacy DynamicK fields at first ratio r={dynamic_topk_ratios[0]:.2f}.",
            "",
            "## Plot Guide",
            "",
            "### What each plotted metric means",
            "",
            "- `Hit@K`: per-question gold coverage ratio in Top-K rerank results: `|TopK ∩ Gold| / |Gold|` (range 0~1).",
            "- Top-K is strict: always uses exactly K rows (no `max(K,G)` window expansion).",
            f"- `Hit@DynamicK(r={dynamic_topk_ratios[0]:.2f})`: per-question gold coverage ratio in rows kept by `score >= r * max_score`.",
            "- `Avg Dynamic-K`: average number of kept rows under the dynamic threshold rule above.",
            "",
            "### How metrics are computed",
            "",
            "- Matching is strict chunk-level: rerank preview is mapped to chunk id, then compared to `gold_chunk_ids`.",
            "- Each point in a curve is aggregated over all questions in one run.",
            "- In `metrics_vs_*.png`, Y values are average coverage ratios (0~1).",
            f"- For dynamic-k across all ratios ({dynamic_topk_ratios[0]:.1f}~{dynamic_topk_ratios[-1]:.1f}), use the `Dynamic-K Ratio Breakdown` table above.",
            "",
            "### What each figure file shows",
            "",
            "- `plots/qe_ratio_metrics_by_rerank_candidates.png`: 3 subplots (RC=10/20/30), X is query-expansion ratio, Y is Hit@1/5/8.",
            f"- `plots/qe_ratio_dynamick_r{dynamic_topk_ratios[0]:.2f}_by_rerank_candidates.png`: 3 subplots (RC=10/20/30), left Y is Hit@DynamicK, right Y is Avg Dynamic-K.",
            "- `plots/qe_ratio_heatmap_hit5.png`: heatmap of Hit@5 by (rerank_candidates, query-expansion ratio).",
            "",
            "## Artifacts",
            "",
            f"- Retrieval details in `runs.json` keep full base/rerank rows (details.md shows Top-{top_k}).",
            "- `interaction_summary.csv`: interaction-level aggregate metrics.",
            "- `question_level.csv`: per-question top retrieval signals.",
            "- `details.md`: per-question/per-run variants and rerank Top-k tables.",
            "- `plots/`: focused charts for query-expansion-ratio sweep only.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_qe_ratio_plots(
    run_dir: Path,
    run_summaries: list[dict],
    dynamic_topk_ratio: float,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    valid_runs = [
        r for r in run_summaries
        if int(r.get("exit_code", 1)) == 0 and int(r.get("retrieval_blocks", 0)) > 0
    ]
    if not valid_runs:
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    by_rc: dict[int, list[dict]] = {}
    for row in valid_runs:
        rc = int(row.get("rerank_candidates", 0))
        by_rc.setdefault(rc, []).append(row)
    rcs = sorted(by_rc.keys())
    if not rcs:
        return []

    # Figure 1: Hit@1/5/8 vs QE ratio for each rerank_candidates
    fig, axes = plt.subplots(1, len(rcs), figsize=(5 * len(rcs), 4.2), sharey=True)
    if len(rcs) == 1:
        axes = [axes]
    for i, rc in enumerate(rcs):
        rows = sorted(by_rc[rc], key=lambda x: float(x.get("translation_ratio", 0.0)))
        xs = [float(r.get("translation_ratio", 0.0)) for r in rows]
        h1 = [float(r.get("avg_hit_at_1", 0.0)) for r in rows]
        h5 = [float(r.get("avg_hit_at_5", 0.0)) for r in rows]
        h8 = [float(r.get("avg_hit_at_8", 0.0)) for r in rows]
        ax = axes[i]
        ax.plot(xs, h1, marker="o", linewidth=1.8, label="Hit@1")
        ax.plot(xs, h5, marker="s", linewidth=1.8, label="Hit@5")
        ax.plot(xs, h8, marker="^", linewidth=1.8, label="Hit@8")
        ax.set_title(f"RC={rc}")
        ax.set_xlabel("Query Expansion Ratio")
        ax.set_ylim(0.6, 1.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Coverage")
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("QE Ratio vs Hit@K by Rerank Candidates", fontsize=12)
    fig.tight_layout()
    out1 = plots_dir / "qe_ratio_metrics_by_rerank_candidates.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    generated.append(str(out1))

    # Figure 2: Hit@DynamicK + Avg Dynamic-K (dual axis)
    fig, axes = plt.subplots(1, len(rcs), figsize=(5 * len(rcs), 4.2), sharey=True)
    if len(rcs) == 1:
        axes = [axes]
    for i, rc in enumerate(rcs):
        rows = sorted(by_rc[rc], key=lambda x: float(x.get("translation_ratio", 0.0)))
        xs = [float(r.get("translation_ratio", 0.0)) for r in rows]
        hit_dyn = [float(r.get("avg_hit_at_dynamic_k", 0.0)) for r in rows]
        avg_k = [float(r.get("avg_dynamic_k", 0.0)) for r in rows]
        ax = axes[i]
        ax2 = ax.twinx()
        l1 = ax.plot(xs, hit_dyn, marker="o", linewidth=1.8, color="#1f77b4", label="Hit@DynamicK")[0]
        l2 = ax2.plot(xs, avg_k, marker="d", linewidth=1.8, color="#ff7f0e", label="Avg Dynamic-K")[0]
        ax.set_title(f"RC={rc}")
        ax.set_xlabel("Query Expansion Ratio")
        ax.set_ylim(0.6, 1.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Hit@DynamicK")
        ax2.set_ylabel("Avg Dynamic-K")
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower right", fontsize=8)
    fig.suptitle(f"QE Ratio vs Dynamic-K Metrics (r={dynamic_topk_ratio:.2f})", fontsize=12)
    fig.tight_layout()
    out2 = plots_dir / f"qe_ratio_dynamick_r{dynamic_topk_ratio:.2f}_by_rerank_candidates.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    generated.append(str(out2))

    # Figure 3: Heatmap for Hit@5
    try:
        import numpy as np
        ratios = sorted({float(r.get("translation_ratio", 0.0)) for r in valid_runs})
        matrix = np.full((len(rcs), len(ratios)), np.nan)
        for i, rc in enumerate(rcs):
            row_map = {float(r.get("translation_ratio", 0.0)): float(r.get("avg_hit_at_5", 0.0)) for r in by_rc[rc]}
            for j, ratio in enumerate(ratios):
                matrix[i, j] = row_map.get(ratio, np.nan)
        fig, ax = plt.subplots(figsize=(max(7, len(ratios) * 0.6), 3.5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.6, vmax=1.0)
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{v:.1f}" for v in ratios], rotation=45, ha="right")
        ax.set_yticks(range(len(rcs)))
        ax.set_yticklabels([str(v) for v in rcs])
        ax.set_xlabel("Query Expansion Ratio")
        ax.set_ylabel("Rerank Candidates")
        ax.set_title("Hit@5 Heatmap")
        fig.colorbar(im, ax=ax, label="Hit@5")
        fig.tight_layout()
        out3 = plots_dir / "qe_ratio_heatmap_hit5.png"
        fig.savefig(out3, dpi=150)
        plt.close(fig)
        generated.append(str(out3))
    except Exception:
        pass

    return generated


def build_dynamic_ratio_plots(
    run_dir: Path,
    question_rows: list[dict],
    dynamic_topk_ratios: list[float],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    if not question_rows or not dynamic_topk_ratios:
        return []

    def _parse_dynamic_map(raw: object) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    ratio_x = list(dynamic_topk_ratios)
    ratio_pos = list(range(len(ratio_x)))
    hit_rates: list[float] = []
    avg_ks: list[float] = []
    row_count = len(question_rows)
    for ratio in ratio_x:
        key = f"{ratio:.4f}"
        hit_sum = 0.0
        k_sum = 0.0
        for row in question_rows:
            dyn = _parse_dynamic_map(row.get("dynamic_by_ratio", {})).get(key, {})
            if isinstance(dyn, dict):
                hit_sum += float(dyn.get("hit", 0.0))
                k_sum += float(dyn.get("k", 0.0))
        hit_rates.append(hit_sum / max(row_count, 1))
        avg_ks.append(k_sum / max(row_count, 1))

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    line = ax.plot(ratio_pos, hit_rates, marker="o", linewidth=2, label="Dynamic-K Accuracy")[0]
    for x_val, y_val, k_val, r_val in zip(ratio_pos, hit_rates, avg_ks, ratio_x):
        ax.annotate(
            f"r={r_val:.2f}\nK={k_val:.2f}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=line.get_color(),
        )
    ax.set_title("Dynamic Ratio vs Accuracy")
    ax.set_xlabel("Dynamic ratio r (non-linear spaced)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(ratio_pos)
    ax.set_xticklabels([f"{r:.2f}" for r in ratio_x], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out1 = plots_dir / "dynamic_ratio_vs_accuracy.png"
    fig.tight_layout()
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    generated.append(str(out1))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    line = ax.plot(
        ratio_pos,
        hit_rates,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Dynamic-K Accuracy",
    )[0]
    for x_val, y_val, k_val, r_val in zip(ratio_pos, hit_rates, avg_ks, ratio_x):
        ax.annotate(
            f"r={r_val:.2f}\nK={k_val:.2f}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=line.get_color(),
        )
    for k in range(1, 9):
        key = f"hit_at_{k}"
        if not any(key in row for row in question_rows):
            continue
        acc = sum(float(row.get(key, 0.0)) for row in question_rows) / max(row_count, 1)
        ax.axhline(acc, linestyle="--", linewidth=1.1, alpha=0.75, label=f"Top-{k}={acc:.2f}")
    ax.set_title("Dynamic Ratio vs Accuracy with Fixed Top-K Baselines")
    ax.set_xlabel("Dynamic ratio r (non-linear spaced)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(ratio_pos)
    ax.set_xticklabels([f"{r:.2f}" for r in ratio_x], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    out2 = plots_dir / "dynamic_ratio_vs_accuracy_with_topk_baselines.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    generated.append(str(out2))
    return generated


def build_markdown_details(
    run_dir: Path,
    questions: list[str],
    run_payloads: list[dict],
    question_rows: list[dict],
    dynamic_topk_ratios: list[float],
    top_k: int,
) -> None:
    lines = [
        "# Retrieval Interaction Details",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Rerank rows shown per run/question: Top-{top_k}",
        "",
    ]

    sorted_runs = sorted(
        run_payloads,
        key=lambda r: (
            r["sweep_group"],
            r["sweep_value"],
            r["variant_mode"],
            r["weight_bm25"],
            r["secondary_variant_weight"],
        ),
    )

    for q_idx, question in enumerate(questions, start=1):
        lines.append(f"## Q{q_idx}")
        lines.append("")
        lines.append(f"Question: `{_md_escape(question)}`")
        lines.append("")

        lines.append("Dynamic-K summary by ratio:")
        ratio_headers = [f"r={ratio:.2f}" for ratio in dynamic_topk_ratios]
        lines.append("| Run ID | " + " | ".join([f"{h} (K/Hit%)" for h in ratio_headers]) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(ratio_headers)) + " |")
        q_rows = [r for r in question_rows if int(r.get("q_id", 0)) == q_idx]
        q_rows_by_run = {str(r.get("run_id", "")): r for r in q_rows}
        for run in sorted_runs:
            row = q_rows_by_run.get(str(run.get("run_id", "")), {})
            dyn_raw = row.get("dynamic_by_ratio", {})
            if isinstance(dyn_raw, str):
                try:
                    dyn_map = json.loads(dyn_raw)
                except Exception:
                    dyn_map = {}
            elif isinstance(dyn_raw, dict):
                dyn_map = dyn_raw
            else:
                dyn_map = {}
            cells = []
            for ratio in dynamic_topk_ratios:
                key = f"{ratio:.4f}"
                dyn = dyn_map.get(key, {})
                k_val = int(dyn.get("k", 0)) if isinstance(dyn, dict) else 0
                hit_val = int(dyn.get("hit", 0)) if isinstance(dyn, dict) else 0
                hit_pct = 100.0 if hit_val == 1 else 0.0
                cells.append(f"{k_val}/{'✅' if hit_val == 1 else '❌'} ({hit_pct:.0f}%)")
            lines.append(f"| {_md_escape(str(run.get('run_id', '')))} | " + " | ".join(cells) + " |")
        lines.append("")

        for run in sorted_runs:
            retrieval_blocks = run.get("retrieval_blocks", [])
            variant_events = run.get("variant_events", [])
            block = retrieval_blocks[q_idx - 1] if q_idx - 1 < len(retrieval_blocks) else {}
            event = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else {}
            variants = event.get("variants", [])
            rerank_rows = block.get("rerank_rows", [])[:top_k]

            lines.append(
                f"### {run['run_id']} "
                f"(sweep={run['sweep_group']}={run['sweep_value']:.2f}, mode={run['variant_mode']}, "
                f"bm25={run['weight_bm25']:.2f}, tr={run['translation_ratio']:.2f}, "
                f"alpha={run['rerank_alpha']:.2f}, svw={run['secondary_variant_weight']:.2f})"
            )
            lines.append("")
            lines.append(
                f"- Weight ratio (vec:bm25): "
                f"`{run['weight_vec']:.2f}:{run['weight_bm25']:.2f}`"
            )
            lines.append(
                f"- Secondary variant weight: `{run['secondary_variant_weight']:.2f}`"
            )
            lines.append("")
            lines.append("Variants:")
            if variants:
                for idx, item in enumerate(variants, start=1):
                    lines.append(f"{idx}. `{_md_escape(item)}`")
            else:
                lines.append("1. `_No variant captured_`")
            lines.append("")
            lines.append(f"Rerank Top-{top_k}:")
            lines.append("| Rank | Score | Source | Chunk Type | Preview |")
            lines.append("| ---: | ---: | --- | --- | --- |")
            if rerank_rows:
                for row in rerank_rows:
                    lines.append(
                        f"| {row['rank']} | {row['score']:.4f} | {_md_escape(row['source'])} | "
                        f"{_md_escape(row['chunk_type'])} | {_md_escape(row['preview'])} |"
                    )
            else:
                lines.append("| - | - | - | - | _No rerank rows captured_ |")
            lines.append("")

    (run_dir / "details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_line_plots(
    run_dir: Path,
    run_summaries: list[dict],
    question_rows: list[dict],
    dynamic_topk_ratio: float,
    dynamic_topk_ratios: list[float],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_lines = [
        ("avg_hit_at_1", "Hit@1"),
        ("avg_hit_at_5", "Hit@5"),
        ("avg_hit_at_8", "Hit@8"),
        ("avg_hit_at_dynamic_k", f"Hit@DynamicK(r={dynamic_topk_ratio:.2f})"),
    ]
    sweep_titles = {
        "bm25_ratio": ("BM25 Ratio", "bm25_ratio"),
        "translation_ratio": ("Translation Ratio", "translation_ratio"),
        "rerank_alpha": ("Rerank Alpha", "rerank_alpha"),
    }

    generated = []
    
    # Get all unique rerank_candidates values for grouping
    rc_values = sorted({int(r.get("rerank_candidates", 0)) for r in run_summaries if r.get("rerank_candidates")})
    
    for sweep_group, (xlabel, slug) in sweep_titles.items():
        rows = [r for r in run_summaries if r.get("sweep_group") == sweep_group]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r.get("sweep_value", 0.0)))

        # One chart per sweep with four metric lines (aggregated across all rerank_candidates).
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [float(r.get("sweep_value", 0.0)) for r in rows]
        for metric_idx, (metric_key, label) in enumerate(metric_lines):
            ys = [
                (float(r.get(metric_key, 0.0)) / max(int(r.get("questions", 0)), 1))
                for r in rows
            ]
            line = ax.plot(xs, ys, marker="o", label=label)[0]
            for x_val, y_val in zip(xs, ys):
                ax.annotate(
                    f"{y_val:.2f}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 6 + metric_idx * 2),
                    ha="center",
                    fontsize=7,
                    color=line.get_color(),
                )
        ax.set_title(f"Retrieval Metrics vs {xlabel} (All Rerank Candidates)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rate")
        ax.set_ylim(0.6, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"metrics_vs_{slug}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))
        
        # Composite big-figure: merge RC-specific metric plots into one canvas.
        if rc_values and len(rc_values) > 1:
            num_rc = len(rc_values)
            cols = min(3, num_rc)
            rows_count = (num_rc + cols - 1) // cols
            fig, axes = plt.subplots(rows_count, cols, figsize=(6 * cols, 5 * rows_count))
            axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

            for idx, rc in enumerate(rc_values):
                ax = axes_list[idx]
                rc_rows = sorted(
                    [r for r in rows if int(r.get("rerank_candidates", 0)) == rc],
                    key=lambda r: float(r.get("sweep_value", 0.0)),
                )
                if not rc_rows:
                    ax.set_visible(False)
                    continue

                xs_rc = [float(r.get("sweep_value", 0.0)) for r in rc_rows]
                for metric_idx, (metric_key, label) in enumerate(metric_lines):
                    ys_rc = [
                        (float(r.get(metric_key, 0.0)) / max(int(r.get("questions", 0)), 1))
                        for r in rc_rows
                    ]
                    line = ax.plot(xs_rc, ys_rc, marker="o", label=label, linewidth=1.5)[0]
                    for x_val, y_val in zip(xs_rc, ys_rc):
                        ax.annotate(
                            f"{y_val:.2f}",
                            (x_val, y_val),
                            textcoords="offset points",
                            xytext=(0, 6 + metric_idx * 1.5),
                            ha="center",
                            fontsize=6,
                            color=line.get_color(),
                        )
                ax.set_title(f"RC = {rc}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Rate")
                ax.set_ylim(0.6, 1.0)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=7)

            for idx in range(num_rc, len(axes_list)):
                axes_list[idx].set_visible(False)

            fig.suptitle(f"Retrieval Metrics vs {xlabel} (All Rerank Candidates)", fontsize=14, fontweight="bold")
            out = plots_dir / f"metrics_vs_{slug}_all_ratios.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            generated.append(str(out))

        fig, ax = plt.subplots(figsize=(9, 5))
        ys_dyn = [float(r.get("avg_dynamic_k", 0.0)) for r in rows]
        line = ax.plot(xs, ys_dyn, marker="o", label="Avg Dynamic-K")[0]
        for x_val, y_val in zip(xs, ys_dyn):
            ax.annotate(
                f"{y_val:.2f}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=line.get_color(),
            )
        ax.set_title(f"Average Dynamic-K vs {xlabel} (All Rerank Candidates) (r={dynamic_topk_ratio:.2f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Dynamic-K")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"avg_dynamic_k_vs_{slug}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))
        
        # Composite big-figure: merge RC-specific Avg Dynamic-K plots into one canvas.
        if rc_values and len(rc_values) > 1:
            num_rc = len(rc_values)
            cols = min(3, num_rc)
            rows_count = (num_rc + cols - 1) // cols
            fig, axes = plt.subplots(rows_count, cols, figsize=(6 * cols, 5 * rows_count))
            axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

            for idx, rc in enumerate(rc_values):
                ax = axes_list[idx]
                rc_rows = sorted(
                    [r for r in rows if int(r.get("rerank_candidates", 0)) == rc],
                    key=lambda r: float(r.get("sweep_value", 0.0)),
                )
                if not rc_rows:
                    ax.set_visible(False)
                    continue

                xs_rc = [float(r.get("sweep_value", 0.0)) for r in rc_rows]
                ys_dyn_rc = [float(r.get("avg_dynamic_k", 0.0)) for r in rc_rows]
                line = ax.plot(
                    xs_rc,
                    ys_dyn_rc,
                    marker="o",
                    label=f"RC={rc}",
                    linewidth=2,
                    color=colors[idx % len(colors)],
                )[0]
                for x_val, y_val in zip(xs_rc, ys_dyn_rc):
                    ax.annotate(
                        f"{y_val:.2f}",
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                        color=line.get_color(),
                    )
                ax.set_title(f"RC = {rc}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Average Dynamic-K")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)

            for idx in range(num_rc, len(axes_list)):
                axes_list[idx].set_visible(False)

            fig.suptitle(f"Average Dynamic-K vs {xlabel} (All Rerank Candidates)", fontsize=14, fontweight="bold")
            out = plots_dir / f"avg_dynamic_k_vs_{slug}_all_ratios.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            generated.append(str(out))
        
        # Generate separate plots for each rerank_candidates value
        if rc_values:
            for rc in rc_values:
                rc_rows = sorted(
                    [r for r in rows if int(r.get("rerank_candidates", 0)) == rc],
                    key=lambda r: float(r.get("sweep_value", 0.0))
                )
                if not rc_rows:
                    continue
                
                fig, ax = plt.subplots(figsize=(9, 5))
                xs_rc = [float(r.get("sweep_value", 0.0)) for r in rc_rows]
                for metric_idx, (metric_key, label) in enumerate(metric_lines):
                    ys_rc = [
                        (float(r.get(metric_key, 0.0)) / max(int(r.get("questions", 0)), 1))
                        for r in rc_rows
                    ]
                    line = ax.plot(xs_rc, ys_rc, marker="o", label=label)[0]
                    for x_val, y_val in zip(xs_rc, ys_rc):
                        ax.annotate(
                            f"{y_val:.2f}",
                            (x_val, y_val),
                            textcoords="offset points",
                            xytext=(0, 6 + metric_idx * 2),
                            ha="center",
                            fontsize=7,
                            color=line.get_color(),
                        )
                ax.set_title(f"Retrieval Metrics vs {xlabel} (RC={rc})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Rate")
                ax.set_ylim(0.6, 1.0)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                out = plots_dir / f"metrics_vs_{slug}_rc{rc}.png"
                fig.tight_layout()
                fig.savefig(out, dpi=150)
                plt.close(fig)
                generated.append(str(out))
                
                fig, ax = plt.subplots(figsize=(9, 5))
                ys_dyn_rc = [float(r.get("avg_dynamic_k", 0.0)) for r in rc_rows]
                line = ax.plot(xs_rc, ys_dyn_rc, marker="o", label="Avg Dynamic-K")[0]
                for x_val, y_val in zip(xs_rc, ys_dyn_rc):
                    ax.annotate(
                        f"{y_val:.2f}",
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=8,
                        color=line.get_color(),
                    )
                ax.set_title(f"Average Dynamic-K vs {xlabel} (RC={rc}) (r={dynamic_topk_ratio:.2f})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Average Dynamic-K")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                out = plots_dir / f"avg_dynamic_k_vs_{slug}_rc{rc}.png"
                fig.tight_layout()
                fig.savefig(out, dpi=150)
                plt.close(fig)
                generated.append(str(out))

    # New: analyze effect of number of rerank candidates if present
    rc_values = sorted({int(r.get("rerank_candidates", 0)) for r in run_summaries if r.get("rerank_candidates")})
    if rc_values and len(rc_values) > 1:
        xs = rc_values
        
        # Composite plot 1: All Hit@X metrics vs rerank_candidates
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric_idx, (metric_key, label) in enumerate(metric_lines):
            ys = []
            for rc in xs:
                rows = [r for r in run_summaries if int(r.get("rerank_candidates", 0)) == rc]
                if not rows:
                    ys.append(0.0)
                    continue
                val = sum(int(r.get(metric_key, 0)) for r in rows) / max(sum(int(r.get("questions", 0)) for r in rows), 1)
                ys.append(val)
            line = ax.plot(xs, ys, marker="o", label=label, linewidth=2)[0]
            for x_val, y_val in zip(xs, ys):
                ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 6 + metric_idx * 2), ha="center", fontsize=8, color=line.get_color())
        ax.set_title("All Hit@X Metrics vs Rerank Candidates (Composite)")
        ax.set_xlabel("Rerank candidates")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.6, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"metrics_vs_rerank_candidates_composite.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))
        
        # Composite plot 2: Avg Dynamic-K and Hit@Dynamic-K vs rerank_candidates (Multiple Ratios)
        # Create subplots for each dynamic_topk_ratio
        if dynamic_topk_ratios and len(dynamic_topk_ratios) > 1:
            num_ratios = len(dynamic_topk_ratios)
            cols = min(3, num_ratios)
            rows_count = (num_ratios + cols - 1) // cols
            fig, axes = plt.subplots(rows_count, cols, figsize=(5 * cols, 5 * rows_count))
            if rows_count == 1 and cols == 1:
                axes = [[axes]]
            elif rows_count == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            # Define distinct colors for consistency
            colors_left = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            colors_right = ["#ff7f0e", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
            
            for idx, ratio in enumerate(dynamic_topk_ratios):
                row_idx = idx // cols
                col_idx = idx % cols
                ax = axes[row_idx][col_idx] if rows_count > 1 else axes[col_idx]
                
                # Gather data for this ratio from per-question dynamic_by_ratio,
                # instead of reusing the first-ratio summary fields.
                ratio_key = f"{ratio:.4f}"
                ys_dyn = []
                ys_hit_dyn = []
                for rc in xs:
                    rows_data = [r for r in question_rows if int(r.get("rerank_candidates", 0)) == rc]
                    if not rows_data:
                        ys_dyn.append(0.0)
                        ys_hit_dyn.append(0.0)
                    else:
                        dyn_ks = []
                        dyn_hits = []
                        for row in rows_data:
                            dyn_map = _parse_dynamic_map(row.get("dynamic_by_ratio", {}))
                            dyn = dyn_map.get(ratio_key, {}) if isinstance(dyn_map, dict) else {}
                            if not isinstance(dyn, dict):
                                dyn = {}
                            dyn_ks.append(float(dyn.get("k", 0.0)))
                            dyn_hits.append(int(dyn.get("hit", 0)))
                        ys_dyn.append(sum(dyn_ks) / max(len(dyn_ks), 1))
                        hit_dyn_rate = sum(dyn_hits) / max(len(dyn_hits), 1)
                        ys_hit_dyn.append(hit_dyn_rate)
                
                ax2 = ax.twinx()
                color_left = colors_left[idx % len(colors_left)]
                color_right = colors_right[idx % len(colors_right)]
                
                line1 = ax.plot(xs, ys_dyn, marker="o", label=f"Avg Dynamic-K (r={ratio:.2f})", linewidth=2, color=color_left)[0]
                for x_val, y_val in zip(xs, ys_dyn):
                    ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=color_left)
                
                line2 = ax2.plot(xs, ys_hit_dyn, marker="s", label=f"Hit@DynamicK (r={ratio:.2f})", linewidth=2, color=color_right)[0]
                for x_val, y_val in zip(xs, ys_hit_dyn):
                    ax2.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color=color_right)
                
                ax.set_title(f"Ratio = {ratio:.2f}")
                ax.set_xlabel("Rerank candidates")
                ax.set_ylabel("Avg Dynamic-K", color=color_left, fontsize=9)
                ax2.set_ylabel("Hit@DynamicK", color=color_right, fontsize=9)
                ax.tick_params(axis="y", labelcolor=color_left)
                ax2.tick_params(axis="y", labelcolor=color_right)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.0, max(ys_dyn) * 1.2 if ys_dyn else 10)
                ax2.set_ylim(0.6, 1.0)
                
                lines = [line1, line2]
                labels = [line.get_label() for line in lines]
                ax.legend(lines, labels, loc="upper left", fontsize=8)
            
            # Hide unused subplots
            for idx in range(num_ratios, rows_count * cols):
                row_idx = idx // cols
                col_idx = idx % cols
                axes[row_idx][col_idx].set_visible(False)
            
            fig.suptitle("Dynamic-K Metrics vs Rerank Candidates (Multiple Ratios)", fontsize=14, fontweight='bold', y=0.995)
        else:
            # Fallback to single plot if only one ratio
            fig, ax = plt.subplots(figsize=(10, 6))
            ys_dyn = []
            ys_hit_dyn = []
            for rc in xs:
                rows = [r for r in run_summaries if int(r.get("rerank_candidates", 0)) == rc]
                if not rows:
                    ys_dyn.append(0.0)
                    ys_hit_dyn.append(0.0)
                else:
                    ys_dyn.append(sum(float(r.get("avg_dynamic_k", 0.0)) for r in rows) / len(rows))
                    hit_dyn_rate = sum(float(r.get("avg_hit_at_dynamic_k", 0.0)) for r in rows) / len(rows)
                    ys_hit_dyn.append(hit_dyn_rate)
            
            ax2 = ax.twinx()
            line1 = ax.plot(xs, ys_dyn, marker="o", label="Avg Dynamic-K", linewidth=2, color="#1f77b4")[0]
            for x_val, y_val in zip(xs, ys_dyn):
                ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=line1.get_color())
            
            line2 = ax2.plot(xs, ys_hit_dyn, marker="s", label=f"Hit@DynamicK(r={dynamic_topk_ratio:.2f})", linewidth=2, color="#ff7f0e")[0]
            for x_val, y_val in zip(xs, ys_hit_dyn):
                ax2.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=line2.get_color())
            
            ax.set_title("Dynamic-K Metrics vs Rerank Candidates (Composite)")
            ax.set_xlabel("Rerank candidates")
            ax.set_ylabel("Avg Dynamic-K", color="#1f77b4")
            ax2.set_ylabel(f"Hit@DynamicK(r={dynamic_topk_ratio:.2f})", color="#ff7f0e")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, max(ys_dyn) * 1.2 if ys_dyn else 10)
            ax2.set_ylim(0.6, 1.0)
            
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper left", fontsize=8)
        
        out = plots_dir / f"dynamic_k_metrics_vs_rerank_candidates_composite.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))
        
        # Individual metric plots for each rerank_candidates value
        for metric_idx, (metric_key, label) in enumerate(metric_lines):
            ys = []
            for rc in xs:
                rows = [r for r in run_summaries if int(r.get("rerank_candidates", 0)) == rc]
                if not rows:
                    ys.append(0.0)
                    continue
                val = sum(int(r.get(metric_key, 0)) for r in rows) / max(sum(int(r.get("questions", 0)) for r in rows), 1)
                ys.append(val)
            fig, ax = plt.subplots(figsize=(8, 4))
            line = ax.plot(xs, ys, marker="o", label=label)[0]
            for x_val, y_val in zip(xs, ys):
                ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=line.get_color())
            ax.set_title(f"{label} vs Rerank Candidates")
            ax.set_xlabel("Rerank candidates")
            ax.set_ylabel("Rate")
            ax.set_ylim(0.6, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            out = plots_dir / f"metrics_vs_rerank_candidates_{metric_key}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            generated.append(str(out))

        # Avg dynamic k vs rerank candidates
        ys_dyn = []
        for rc in xs:
            rows = [r for r in run_summaries if int(r.get("rerank_candidates", 0)) == rc]
            if not rows:
                ys_dyn.append(0.0)
            else:
                ys_dyn.append(sum(float(r.get("avg_dynamic_k", 0.0)) for r in rows) / len(rows))
        fig, ax = plt.subplots(figsize=(8, 4))
        line = ax.plot(xs, ys_dyn, marker="o", label="Avg Dynamic-K")[0]
        for x_val, y_val in zip(xs, ys_dyn):
            ax.annotate(f"{y_val:.2f}", (x_val, y_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=line.get_color())
        ax.set_title("Average Dynamic-K vs Rerank Candidates")
        ax.set_xlabel("Rerank candidates")
        ax.set_ylabel("Average Dynamic-K")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"avg_dynamic_k_vs_rerank_candidates.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))

    def _parse_dynamic_map(raw: object) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    # Global dynamic-ratio curve aggregated over all (run, question) rows.
    if question_rows and dynamic_topk_ratios:
        ratio_x = list(dynamic_topk_ratios)
        # Non-linear display spacing: treat each ratio as an ordered level to avoid crowding near 0~0.1.
        ratio_pos = list(range(len(ratio_x)))
        hit_rates = []
        avg_ks = []
        row_count = len(question_rows)
        for ratio in ratio_x:
            key = f"{ratio:.4f}"
            hit_sum = 0
            k_sum = 0
            for row in question_rows:
                dyn = _parse_dynamic_map(row.get("dynamic_by_ratio", {})).get(key, {})
                if isinstance(dyn, dict):
                    hit_sum += int(dyn.get("hit", 0))
                    k_sum += int(dyn.get("k", 0))
            hit_rates.append(hit_sum / max(row_count, 1))
            avg_ks.append(k_sum / max(row_count, 1))

        fig, ax = plt.subplots(figsize=(9, 5))
        line = ax.plot(ratio_pos, hit_rates, marker="o", linewidth=2, label="Dynamic-K Accuracy")[0]
        for x_val, y_val, k_val, r_val in zip(ratio_pos, hit_rates, avg_ks, ratio_x):
            ax.annotate(
                f"r={r_val:.2f}\nK={k_val:.2f}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=line.get_color(),
            )
        ax.set_title("Dynamic Ratio vs Accuracy")
        ax.set_xlabel("Dynamic ratio r (non-linear spaced)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.6, 1.0)
        ax.set_xticks(ratio_pos)
        ax.set_xticklabels([f"{r:.2f}" for r in ratio_x], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / "dynamic_ratio_vs_accuracy.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))

        fig, ax = plt.subplots(figsize=(10, 5.5))
        dynamic_color = "#1f77b4"
        baseline_colors = [
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#17becf",
        ]
        line = ax.plot(
            ratio_pos,
            hit_rates,
            marker="o",
            linewidth=2,
            color=dynamic_color,
            label="Dynamic-K Accuracy",
        )[0]
        for x_val, y_val, k_val, r_val in zip(ratio_pos, hit_rates, avg_ks, ratio_x):
            ax.annotate(
                f"r={r_val:.2f}\nK={k_val:.2f}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=line.get_color(),
            )

        for k in range(1, 9):
            key = f"hit_at_{k}"
            if not any(key in row for row in question_rows):
                continue
            acc = sum(int(row.get(key, 0)) for row in question_rows) / max(row_count, 1)
            ax.axhline(
                acc,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color=baseline_colors[(k - 1) % len(baseline_colors)],
                label=f"Top-{k}={acc:.2f}",
            )

        ax.set_title("Dynamic Ratio vs Accuracy with Fixed Top-K Baselines")
        ax.set_xlabel("Dynamic ratio r (non-linear spaced)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.6, 1.0)
        ax.set_xticks(ratio_pos)
        ax.set_xticklabels([f"{r:.2f}" for r in ratio_x], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        out = plots_dir / "dynamic_ratio_vs_accuracy_with_topk_baselines.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))

    return generated


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    doc_file = resolve_path(project_root, args.doc_file).resolve()
    question_file = resolve_path(project_root, args.question_file).resolve()
    gold_file = resolve_path(project_root, args.gold_file).resolve()
    gold_chunks_file = resolve_path(project_root, args.gold_chunks_file).resolve() if args.gold_chunks_file else None
    eval_topks = parse_eval_topks(args.eval_topk)
    max_eval_k = max(eval_topks)
    dynamic_topk_ratios = parse_dynamic_topk_ratios(args.dynamic_topk_ratio)
    dynamic_topk_ratio = dynamic_topk_ratios[0]
    output_base = resolve_path(project_root, args.output_dir).resolve()
    chat_box_path = project_root / "chat_box.py"

    if not doc_file.exists():
        raise FileNotFoundError(f"Doc file not found: {doc_file}")
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if not chat_box_path.exists():
        raise FileNotFoundError(f"chat_box.py not found under project root: {chat_box_path}")

    plan = build_run_plan(args)
    questions = load_questions(question_file)
    if not questions:
        raise ValueError(f"No valid questions loaded from: {question_file}")
    gold_map = load_gold_map(gold_file if gold_file.exists() else None)

    model_name = configured_model_name(project_root / "config.yaml", None)
    ollama_info = ensure_ollama(args, model_name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"retrieval_weight_ablation_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    default_gold_chunks_file = answers_dir / "master_md.chunks.jsonl"
    chunk_catalog = {}
    chunk_catalog_source = ""
    if gold_chunks_file and gold_chunks_file.exists():
        chunk_catalog = load_chunk_catalog(gold_chunks_file)
        chunk_catalog_source = str(gold_chunks_file)

    print(f"[INFO] Project root: {project_root}", flush=True)
    print(f"[INFO] Doc file: {doc_file}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
    print(f"[INFO] Gold file: {gold_file if gold_file.exists() else '[missing]'}", flush=True)
    if gold_chunks_file:
        print(f"[INFO] Gold chunks file: {gold_chunks_file if gold_chunks_file.exists() else '[missing]'}", flush=True)
    else:
        print(f"[INFO] Gold chunks file: [auto] {default_gold_chunks_file}", flush=True)
    print(f"[INFO] Evaluation top-k: {eval_topks}", flush=True)
    print(f"[INFO] Dynamic top-k ratios: {[round(r, 2) for r in dynamic_topk_ratios]}", flush=True)
    print(
        f"[INFO] Fixed non-target values: bm25_ratio={args.fixed_bm25_ratio:.2f}, "
        f"translation_ratio={args.fixed_translation_ratio:.2f}, rerank_alpha={args.fixed_rerank_alpha:.2f}",
        flush=True,
    )
    print(f"[INFO] Interaction runs planned: {len(plan)}", flush=True)
    print(f"[INFO] Output dir: {run_dir}", flush=True)
    print(f"[INFO] Ollama host: {args.ollama_host}", flush=True)
    print(f"[INFO] Ollama ready: {ollama_info['ready']}", flush=True)
    if not ollama_info["ready"]:
        print("[WARN] Ollama is not ready. Runs may fail unless the service is started.", flush=True)

    run_payloads = []
    question_level_rows = []
    interaction_summary_rows = []

    for idx, item in enumerate(plan, start=1):
        run_id_str = item["run_id"]
        log_path = logs_dir / f"{run_id_str}.log"
        answer_path = answers_dir / f"{run_id_str}.answers.txt"
        chunk_path = answers_dir / "master_md.chunks.jsonl"
        cmd = [
            args.python_bin,
            str(chat_box_path),
            "--doc-file", str(doc_file),
            "--question-file", str(question_file),
            "--answer-file", str(answer_path),
            "--save-chunks-file", str(chunk_path),
            "--debug",
            "--weight-vec", str(item["weight_vec"]),
            "--weight-bm25", str(item["weight_bm25"]),
            "--variant-mode", item["variant_mode"],
            "--secondary-variant-weight", str(item["secondary_variant_weight"]),
            "--rerank-alpha", str(item["rerank_alpha"]),
            "--rerank-candidates", str(item.get("rerank_candidates", 30)),
        ]
        optional_overrides = [
            ("--enable-query-expansion", args.enable_query_expansion if args.enable_query_expansion is not None else "true"),
            ("--expansion-model", args.expansion_model),
            ("--expansion-paraphrases", args.expansion_paraphrases),
            ("--expansion-add-translation", args.expansion_add_translation),
            ("--expansion-source-lang", args.expansion_source_lang),
            ("--enable-multi-variant-rerank", args.enable_multi_variant_rerank),
            ("--enable-cache", args.enable_cache),
            ("--enable-memory", args.enable_memory),
        ]
        for flag, value in optional_overrides:
            if value is not None:
                cmd.extend([flag, str(value)])

        print(
            f"[RUN {idx}/{len(plan)}] {run_id_str} "
            f"(vec={item['weight_vec']}, bm25={item['weight_bm25']}, "
            f"mode={item['variant_mode']}, tr={item['translation_ratio']}, "
            f"alpha={item['rerank_alpha']}, svw={item['secondary_variant_weight']}, "
            f"rc={item.get('rerank_candidates', 30)})",
            flush=True,
        )
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
            f"[RUN {idx}/{len(plan)}] exit={proc.returncode} duration={duration_sec:.2f}s",
            flush=True,
        )

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        blocks = extract_retrieval_markdown_blocks(log_text)
        parsed_blocks = [parse_retrieval_block(block) for block in blocks]
        variant_events = extract_query_variant_events(log_text)

        if not chunk_catalog:
            if gold_chunks_file and gold_chunks_file.exists():
                chunk_catalog = load_chunk_catalog(gold_chunks_file)
                chunk_catalog_source = str(gold_chunks_file)
            elif chunk_path.exists():
                chunk_catalog = load_chunk_catalog(chunk_path)
                chunk_catalog_source = str(chunk_path)
            elif default_gold_chunks_file.exists():
                chunk_catalog = load_chunk_catalog(default_gold_chunks_file)
                chunk_catalog_source = str(default_gold_chunks_file)
            if chunk_catalog:
                print(
                    f"[RUN {idx}/{len(plan)}] Loaded strict eval chunk catalog from: {chunk_catalog_source} "
                    f"(chunks={len(chunk_catalog.get('chunks', []))})",
                    flush=True,
                )

        per_run_rows = []
        for q_idx, question in enumerate(questions, start=1):
            lang = detect_question_language(question)
            block = parsed_blocks[q_idx - 1] if q_idx - 1 < len(parsed_blocks) else {"base_rows": [], "rerank_rows": []}
            variant_event = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else {"variants": []}

            base_rows = block.get("base_rows", [])
            rerank_rows = block.get("rerank_rows", [])
            base_top1 = base_rows[0] if base_rows else None
            rerank_top1 = rerank_rows[0] if rerank_rows else None
            rerank_top2 = rerank_rows[1] if len(rerank_rows) > 1 else None
            margin = None
            if rerank_top1 and rerank_top2:
                margin = round(rerank_top1["score"] - rerank_top2["score"], 4)

            gold = gold_map.get(question, {})
            gold_chunk = str(gold.get("gold_chunk", ""))
            gold_chunk_ids = list(gold.get("gold_chunk_ids", []))

            eval_limit = max(max_eval_k, len(gold_chunk_ids))
            eval_rows = rerank_rows[:eval_limit]
            hit_map = {k: 0.0 for k in eval_topks}
            fixed_hit_map = {k: 0.0 for k in range(1, 9)}
            top1_matched_chunk_ids = []
            dynamic_by_ratio = {f"{ratio:.4f}": {"k": 0, "hit": 0.0} for ratio in dynamic_topk_ratios}
            if eval_rows and gold_chunk_ids and chunk_catalog.get("chunks"):
                for rr in eval_rows:
                    rr["matched_chunk_ids"] = match_preview_to_chunk_ids(rr.get("preview", ""), chunk_catalog)
                top1_matched_chunk_ids = eval_rows[0].get("matched_chunk_ids", [])
                gold_chunk_id_set = set(gold_chunk_ids)
                for k in range(1, 9):
                    window_k = k
                    considered = eval_rows[:window_k]
                    covered_chunk_ids = {
                        cid
                        for rr in considered
                        for cid in rr.get("matched_chunk_ids", [])
                        if cid in gold_chunk_id_set
                    }
                    fixed_hit_map[k] = (len(covered_chunk_ids) / len(gold_chunk_id_set)) if gold_chunk_id_set else 0.0
                for k in eval_topks:
                    window_k = k
                    considered = eval_rows[:window_k]
                    covered_chunk_ids = {
                        cid
                        for rr in considered
                        for cid in rr.get("matched_chunk_ids", [])
                        if cid in gold_chunk_id_set
                    }
                    hit_map[k] = (len(covered_chunk_ids) / len(gold_chunk_id_set)) if gold_chunk_id_set else 0.0
                max_score = eval_rows[0].get("score")
                if isinstance(max_score, (int, float)):
                    max_score_f = float(max_score)
                    for ratio in dynamic_topk_ratios:
                        threshold = max_score_f * ratio
                        dynamic_rows = [
                            rr for rr in eval_rows
                            if isinstance(rr.get("score"), (int, float)) and float(rr.get("score")) >= threshold
                        ]
                        dynamic_k = len(dynamic_rows)
                        dynamic_covered_chunk_ids = {
                            cid
                            for rr in dynamic_rows
                            for cid in rr.get("matched_chunk_ids", [])
                            if cid in gold_chunk_id_set
                        }
                        dynamic_hit = (len(dynamic_covered_chunk_ids) / len(gold_chunk_id_set)) if gold_chunk_id_set else 0.0
                        dynamic_by_ratio[f"{ratio:.4f}"] = {"k": dynamic_k, "hit": dynamic_hit}

            first_ratio_key = f"{dynamic_topk_ratios[0]:.4f}"
            first_dynamic = dynamic_by_ratio.get(first_ratio_key, {"k": 0, "hit": 0})

            row = {
                "run_id": run_id_str,
                "sweep_group": item["sweep_group"],
                "sweep_var": item["sweep_var"],
                "sweep_value": item["sweep_value"],
                "variant_mode": item["variant_mode"],
                "secondary_variant_weight": item["secondary_variant_weight"],
                "translation_ratio": item["translation_ratio"],
                "rerank_alpha": item["rerank_alpha"],
                "weight_vec": item["weight_vec"],
                "weight_bm25": item["weight_bm25"],
                "q_id": q_idx,
                "lang": lang,
                "question": question,
                "gold_chunk_ids": "|".join(str(cid) for cid in gold_chunk_ids),
                "gold_chunk": gold_chunk,
                "variants": " || ".join(variant_event.get("variants", [])),
                "base_top1_score": base_top1["score"] if base_top1 else None,
                "base_top1_chunk_type": base_top1["chunk_type"] if base_top1 else "",
                "base_top1_preview": base_top1["preview"] if base_top1 else "",
                "rerank_top1_score": rerank_top1["score"] if rerank_top1 else None,
                "rerank_top1_chunk_type": rerank_top1["chunk_type"] if rerank_top1 else "",
                "rerank_top1_preview": rerank_top1["preview"] if rerank_top1 else "",
                "rerank_margin_top1_top2": margin,
                "top1_matched_chunk_ids": "|".join(str(cid) for cid in top1_matched_chunk_ids),
                "dynamic_k": first_dynamic.get("k", 0),
                "dynamic_ratio": first_ratio_key,
                "hit_at_dynamic_k": first_dynamic.get("hit", 0),
                "dynamic_by_ratio": json.dumps(dynamic_by_ratio, ensure_ascii=False),
            }
            for k in range(1, 9):
                row[f"hit_at_{k}"] = fixed_hit_map.get(k, 0)
            for k in eval_topks:
                row[f"hit_at_{k}"] = hit_map.get(k, row.get(f"hit_at_{k}", 0))
            per_run_rows.append(row)
            question_level_rows.append(row.copy())

        summary = summarize_run_rows(per_run_rows)
        summary.update(
            {
                "run_id": run_id_str,
                "sweep_group": item["sweep_group"],
                "sweep_var": item["sweep_var"],
                "sweep_value": item["sweep_value"],
                "variant_mode": item["variant_mode"],
                "secondary_variant_weight": item["secondary_variant_weight"],
                "translation_ratio": item["translation_ratio"],
                "rerank_alpha": item["rerank_alpha"],
                "weight_vec": item["weight_vec"],
                "weight_bm25": item["weight_bm25"],
                "rerank_candidates": item.get("rerank_candidates"),
                "exit_code": proc.returncode,
                "duration_sec": round(duration_sec, 3),
                "retrieval_blocks": len(blocks),
                "questions_expected": len(questions),
            }
        )
        interaction_summary_rows.append(summary)

        run_payloads.append(
            {
                **item,
                "exit_code": proc.returncode,
                "duration_sec": round(duration_sec, 3),
                "log_path": str(log_path),
                "answer_path": str(answer_path),
                "retrieval_blocks": parsed_blocks,
                "variant_events": variant_events,
                "summary": summary,
            }
        )

    # Backward-compatible aliases: keep *_rate equal to averaged coverage ratio.
    for summary in interaction_summary_rows:
        summary["hit_at_1_rate"] = float(summary.get("avg_hit_at_1", 0.0))
        summary["hit_at_5_rate"] = float(summary.get("avg_hit_at_5", 0.0))
        summary["hit_at_8_rate"] = float(summary.get("avg_hit_at_8", 0.0))
        summary["hit_at_dynamic_k_rate"] = float(summary.get("avg_hit_at_dynamic_k", 0.0))

    write_json(
        run_dir / "runs.json",
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_file": str(doc_file),
            "question_file": str(question_file),
            "gold_file": str(gold_file),
            "gold_chunks_file": chunk_catalog_source,
            "eval_topks": eval_topks,
            "dynamic_topk_ratios": dynamic_topk_ratios,
            "dynamic_topk_ratio": dynamic_topk_ratio,
            "bm25_ratios": _validate_ratio_list("bm25-ratios", args.bm25_ratios),
            "translation_ratios": _validate_ratio_list("translation-ratios", args.translation_ratios),
            "rerank_alphas": _validate_ratio_list("rerank-alphas", args.rerank_alphas),
            "fixed_bm25_ratio": args.fixed_bm25_ratio,
            "fixed_translation_ratio": args.fixed_translation_ratio,
            "fixed_rerank_alpha": args.fixed_rerank_alpha,
            "top_k": args.top_k,
            "runs": run_payloads,
        },
    )

    write_csv(
        run_dir / "question_level.csv",
        question_level_rows,
        [
            "run_id",
            "sweep_group",
            "sweep_var",
            "sweep_value",
            "variant_mode",
            "secondary_variant_weight",
            "translation_ratio",
            "rerank_alpha",
            "weight_vec",
            "weight_bm25",
            "q_id",
            "lang",
            "question",
            "gold_chunk_ids",
            "gold_chunk",
            "variants",
            "base_top1_score",
            "base_top1_chunk_type",
            "base_top1_preview",
            "rerank_top1_score",
            "rerank_top1_chunk_type",
            "rerank_top1_preview",
            "rerank_margin_top1_top2",
            "top1_matched_chunk_ids",
            "hit_at_1",
            "hit_at_2",
            "hit_at_3",
            "hit_at_4",
            "hit_at_5",
            "hit_at_6",
            "hit_at_7",
            "hit_at_8",
            "dynamic_k",
            "dynamic_ratio",
            "hit_at_dynamic_k",
            "dynamic_by_ratio",
        ],
    )

    write_csv(
        run_dir / "interaction_summary.csv",
        interaction_summary_rows,
        [
            "run_id",
            "sweep_group",
            "sweep_var",
            "sweep_value",
            "variant_mode",
            "secondary_variant_weight",
            "translation_ratio",
            "rerank_alpha",
            "weight_vec",
            "weight_bm25",
            "rerank_candidates",
            "exit_code",
            "duration_sec",
            "questions_expected",
            "retrieval_blocks",
            "questions",
            "avg_base_top1_score",
            "avg_rerank_top1_score",
            "avg_rerank_margin_top1_top2",
            "unique_rerank_top1_chunks",
            "avg_hit_at_1",
            "hit_at_1_rate",
            "avg_hit_at_5",
            "hit_at_5_rate",
            "avg_hit_at_8",
            "hit_at_8_rate",
            "avg_hit_at_dynamic_k",
            "hit_at_dynamic_k_rate",
            "avg_dynamic_k",
        ],
    )

    build_markdown_summary(
        run_dir=run_dir,
        doc_file=doc_file,
        question_file=question_file,
        run_summaries=interaction_summary_rows,
        question_rows=question_level_rows,
        top_k=args.top_k,
        eval_topks=eval_topks,
        dynamic_topk_ratios=dynamic_topk_ratios,
    )
    build_markdown_details(
        run_dir=run_dir,
        questions=questions,
        run_payloads=run_payloads,
        question_rows=question_level_rows,
        dynamic_topk_ratios=dynamic_topk_ratios,
        top_k=args.top_k,
    )
    plot_files = build_qe_ratio_plots(
        run_dir=run_dir,
        run_summaries=interaction_summary_rows,
        dynamic_topk_ratio=dynamic_topk_ratio,
    )
    plot_files.extend(
        build_dynamic_ratio_plots(
            run_dir=run_dir,
            question_rows=question_level_rows,
            dynamic_topk_ratios=dynamic_topk_ratios,
        )
    )

    print("[INFO] Interaction ablation finished.", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Details: {run_dir / 'details.md'}", flush=True)
    print(f"[INFO] Interaction table: {run_dir / 'interaction_summary.csv'}", flush=True)
    print(f"[INFO] Question-level table: {run_dir / 'question_level.csv'}", flush=True)
    if plot_files:
        print(f"[INFO] Plots generated: {len(plot_files)} under {run_dir / 'plots'}", flush=True)
    else:
        print("[WARN] Plots were not generated (matplotlib may be unavailable).", flush=True)


if __name__ == "__main__":
    main()
