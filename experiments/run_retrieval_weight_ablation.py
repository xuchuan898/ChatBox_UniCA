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

DEFAULT_SWEEP_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval-focused single-variable ablations on one document: "
            "bm25:vector ratio sweep, retrieval translation ratio sweep, and rerank alpha sweep. "
            "Non-target variables are fixed to 0.5 by default."
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
        default=DEFAULT_SWEEP_VALUES,
        help="BM25 ratio values in [0,1] for bm25:vector ratio sweep (w_vec=1-r, w_bm25=r).",
    )
    parser.add_argument(
        "--translation-ratios",
        type=float,
        nargs="*",
        default=DEFAULT_SWEEP_VALUES,
        help="Retrieval-side translation ratio values in [0,1] (secondary_variant_weight sweep).",
    )
    parser.add_argument(
        "--rerank-alphas",
        type=float,
        nargs="*",
        default=DEFAULT_SWEEP_VALUES,
        help="Rerank translation fusion alpha values in [0,1] (alpha*src + (1-alpha)*translated).",
    )
    parser.add_argument(
        "--fixed-bm25-ratio",
        type=float,
        default=0.5,
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
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help=(
            "Comma-separated dynamic top-k ratios. "
            "For each ratio r, keep candidates with score >= r * max_score."
        ),
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        nargs="*",
        default=[30],
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

    plan = []
    # A) Sweep bm25:vector ratio, fix retrieval translation ratio + rerank alpha.
    for bm25_ratio in bm25_ratios:
        tr = fixed_translation_ratio
        alpha = fixed_rerank_alpha
        svw = 1.0 if args.variant_mode == "primary_only" else tr
        for rc in rerank_candidates:
            run_id = (
                f"bm25_ratio__r{_weight_slug(bm25_ratio)}"
                f"__tr{_weight_slug(tr)}__ra{_weight_slug(alpha)}__rc{rc}"
            )
            plan.append(
                {
                    "run_id": run_id,
                    "sweep_group": "bm25_ratio",
                    "sweep_var": "bm25_ratio",
                    "sweep_value": float(bm25_ratio),
                    "weight_vec": float(1.0 - bm25_ratio),
                    "weight_bm25": float(bm25_ratio),
                    "variant_mode": args.variant_mode,
                    "secondary_variant_weight": float(svw),
                    "translation_ratio": float(tr),
                    "rerank_alpha": float(alpha),
                    "rerank_candidates": int(rc),
                }
            )

    # B) Sweep retrieval translation ratio, fix bm25 ratio + rerank alpha.
    for tr in translation_ratios:
        bm25_ratio = fixed_bm25_ratio
        alpha = fixed_rerank_alpha
        svw = 1.0 if args.variant_mode == "primary_only" else tr
        for rc in rerank_candidates:
            run_id = (
                f"translation_ratio__r{_weight_slug(tr)}"
                f"__bm25{_weight_slug(bm25_ratio)}__ra{_weight_slug(alpha)}__rc{rc}"
            )
            plan.append(
                {
                    "run_id": run_id,
                    "sweep_group": "translation_ratio",
                    "sweep_var": "translation_ratio",
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

    # C) Sweep rerank translation alpha, fix bm25 ratio + retrieval translation ratio.
    for alpha in rerank_alphas:
        bm25_ratio = fixed_bm25_ratio
        tr = fixed_translation_ratio
        svw = 1.0 if args.variant_mode == "primary_only" else tr
        for rc in rerank_candidates:
            run_id = (
                f"rerank_alpha__r{_weight_slug(alpha)}"
                f"__bm25{_weight_slug(bm25_ratio)}__tr{_weight_slug(tr)}__rc{rc}"
            )
            plan.append(
                {
                    "run_id": run_id,
                    "sweep_group": "rerank_alpha",
                    "sweep_var": "rerank_alpha",
                    "sweep_value": float(alpha),
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
        "hit_at_1_count": sum(int(r.get("hit_at_1", 0)) for r in rows),
        "hit_at_5_count": sum(int(r.get("hit_at_5", 0)) for r in rows),
        "hit_at_8_count": sum(int(r.get("hit_at_8", 0)) for r in rows),
        "hit_at_dynamic_k_count": sum(int(r.get("hit_at_dynamic_k", 0)) for r in rows),
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
        "| Run ID | Sweep | Value | Variant Mode | w_vec | w_bm25 | Translation Ratio | Rerank Alpha | Hit@1 | Hit@5 | Hit@8 | Hit@DynamicK | Avg Dynamic-K | Avg Rerank Top1 | Avg Margin(1-2) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_summaries:
        q = max(int(row.get("questions", 0)), 1)
        h1 = int(row.get("hit_at_1_count", 0))
        h5 = int(row.get("hit_at_5_count", 0))
        h8 = int(row.get("hit_at_8_count", 0))
        hd = int(row.get("hit_at_dynamic_k_count", 0))
        h1_pct = (h1 / q) * 100.0
        h5_pct = (h5 / q) * 100.0
        h8_pct = (h8 / q) * 100.0
        hd_pct = (hd / q) * 100.0
        lines.append(
            f"| {row['run_id']} | {row['sweep_group']} | {row['sweep_value']:.2f} | {row['variant_mode']} | "
            f"{row['weight_vec']:.2f} | {row['weight_bm25']:.2f} | {row['translation_ratio']:.2f} | {row['rerank_alpha']:.2f} | {row.get('rerank_candidates', '')} | "
            f"{h1}/{q} ({h1_pct:.2f}%) | {h5}/{q} ({h5_pct:.2f}%) | {h8}/{q} ({h8_pct:.2f}%) | "
            f"{hd}/{q} ({hd_pct:.2f}%) | {row.get('avg_dynamic_k', 0.0):.2f} | "
            f"{row['avg_rerank_top1_score']:.4f} | {row['avg_rerank_margin_top1_top2']:.4f} |"
        )

    lines.extend(["", "## Dynamic-K Ratio Breakdown", ""])
    dyn_headers = []
    for ratio in dynamic_topk_ratios:
        dyn_headers.extend([f"r={ratio:.2f} Hit", f"r={ratio:.2f} AvgK"])
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
            hit_count = 0
            sum_k = 0
            for qrow in run_q_rows:
                dyn_map = _parse_dynamic_map(qrow.get("dynamic_by_ratio", {}))
                dyn = dyn_map.get(key, {})
                if isinstance(dyn, dict):
                    hit_count += int(dyn.get("hit", 0))
                    sum_k += int(dyn.get("k", 0))
            hit_pct = (hit_count / q) * 100.0
            avg_k = (sum_k / q) if q else 0.0
            cells.append(f"{hit_count}/{q} ({hit_pct:.2f}%)")
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
            f"- `Hit@1`: number of questions where all `gold_chunk_ids` are covered in top-W rerank results with W=max(1, G), G=len(`gold_chunk_ids`) (`hit_at_1_count`, out of `Q`).",
            f"- `Hit@5`: number of questions where all `gold_chunk_ids` are covered in top-W rerank results with W=max(5, G), G=len(`gold_chunk_ids`) (`hit_at_5_count`, out of `Q`).",
            f"- `Hit@8`: number of questions where all `gold_chunk_ids` are covered in top-W rerank results with W=max(8, G), G=len(`gold_chunk_ids`) (`hit_at_8_count`, out of `Q`).",
            f"- `Hit@DynamicK(r={dynamic_topk_ratios[0]:.2f})`: for each question, keep rerank rows with `score >= r * max_score`; count as hit only when all `gold_chunk_ids` are covered.",
            "- `Avg Dynamic-K`: average number of kept rows under the dynamic threshold rule above.",
            "",
            "### How metrics are computed",
            "",
            "- Matching is strict chunk-level: rerank preview is mapped to chunk id, then compared to `gold_chunk_ids`.",
            "- Each point in a curve is aggregated over all questions in one run.",
            "- In `metrics_vs_*.png`, Y values are hit counts (not percentages). Convert to accuracy by `count / Q`.",
            f"- For dynamic-k across all ratios ({dynamic_topk_ratios[0]:.1f}~{dynamic_topk_ratios[-1]:.1f}), use the `Dynamic-K Ratio Breakdown` table above.",
            "",
            "### What each figure file shows",
            "",
            "- `plots/metrics_vs_bm25_ratio.png`: X is BM25 ratio sweep (`w_bm25`), with other variables fixed.",
            "- `plots/metrics_vs_translation_ratio.png`: X is retrieval translation ratio sweep.",
            "- `plots/metrics_vs_rerank_alpha.png`: X is rerank fusion alpha sweep.",
            "- `plots/avg_dynamic_k_vs_bm25_ratio.png`: X is BM25 ratio; Y is average dynamic-K.",
            "- `plots/avg_dynamic_k_vs_translation_ratio.png`: X is translation ratio; Y is average dynamic-K.",
            "- `plots/avg_dynamic_k_vs_rerank_alpha.png`: X is rerank alpha; Y is average dynamic-K.",
            "- `plots/metrics_vs_rerank_candidates_*.png`: X is number of candidates sent to rerank; Y is metric rate.",
            "- `plots/avg_dynamic_k_vs_rerank_candidates.png`: X is number of candidates sent to rerank; Y is average dynamic-K.",
            "",
            "## Artifacts",
            "",
            f"- Retrieval details in `runs.json` keep full base/rerank rows (details.md shows Top-{top_k}).",
            "- `interaction_summary.csv`: interaction-level aggregate metrics.",
            "- `question_level.csv`: per-question top retrieval signals.",
            "- `details.md`: per-question/per-run variants and rerank Top-k tables.",
            "- `plots/`: line charts for bm25_ratio / translation_ratio / rerank_alpha sweeps.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        ("hit_at_1_count", "Hit@1"),
        ("hit_at_5_count", "Hit@5"),
        ("hit_at_8_count", "Hit@8"),
        ("hit_at_dynamic_k_count", f"Hit@DynamicK(r={dynamic_topk_ratio:.2f})"),
    ]
    sweep_titles = {
        "bm25_ratio": ("BM25 Ratio", "bm25_ratio"),
        "translation_ratio": ("Translation Ratio", "translation_ratio"),
        "rerank_alpha": ("Rerank Alpha", "rerank_alpha"),
    }

    generated = []
    for sweep_group, (xlabel, slug) in sweep_titles.items():
        rows = [r for r in run_summaries if r.get("sweep_group") == sweep_group]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r.get("sweep_value", 0.0)))

        # One chart per sweep with four metric lines.
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
        ax.set_title(f"Retrieval Metrics vs {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"metrics_vs_{slug}.png"
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
        ax.set_title(f"Average Dynamic-K vs {xlabel} (r={dynamic_topk_ratio:.2f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Dynamic-K")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out = plots_dir / f"avg_dynamic_k_vs_{slug}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        generated.append(str(out))

    # New: analyze effect of number of rerank candidates if present
    rc_values = sorted({int(r.get("rerank_candidates", 0)) for r in run_summaries if r.get("rerank_candidates")})
    if rc_values:
        xs = rc_values
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
            ax.set_ylim(0.0, 1.05)
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
        line = ax.plot(ratio_x, hit_rates, marker="o", linewidth=2, label="Dynamic-K Accuracy")[0]
        for x_val, y_val, k_val in zip(ratio_x, hit_rates, avg_ks):
            ax.annotate(
                f"r={x_val:.2f}\nK={k_val:.2f}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=line.get_color(),
            )
        ax.set_title("Dynamic Ratio vs Accuracy")
        ax.set_xlabel("Dynamic ratio r")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(ratio_x)
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
            ratio_x,
            hit_rates,
            marker="o",
            linewidth=2,
            color=dynamic_color,
            label="Dynamic-K Accuracy",
        )[0]
        for x_val, y_val, k_val in zip(ratio_x, hit_rates, avg_ks):
            ax.annotate(
                f"r={x_val:.2f}\nK={k_val:.2f}",
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
        ax.set_xlabel("Dynamic ratio r")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(ratio_x)
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

    ollama_info = ensure_ollama(args)
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
            hit_map = {k: 0 for k in eval_topks}
            fixed_hit_map = {k: 0 for k in range(1, 9)}
            top1_matched_chunk_ids = []
            dynamic_by_ratio = {f"{ratio:.4f}": {"k": 0, "hit": 0} for ratio in dynamic_topk_ratios}
            if eval_rows and gold_chunk_ids and chunk_catalog.get("chunks"):
                for rr in eval_rows:
                    rr["matched_chunk_ids"] = match_preview_to_chunk_ids(rr.get("preview", ""), chunk_catalog)
                top1_matched_chunk_ids = eval_rows[0].get("matched_chunk_ids", [])
                gold_chunk_id_set = set(gold_chunk_ids)
                for k in range(1, 9):
                    window_k = max(k, len(gold_chunk_id_set))
                    considered = eval_rows[:window_k]
                    covered_chunk_ids = {
                        cid
                        for rr in considered
                        for cid in rr.get("matched_chunk_ids", [])
                        if cid in gold_chunk_id_set
                    }
                    fixed_hit_map[k] = 1 if gold_chunk_id_set.issubset(covered_chunk_ids) else 0
                for k in eval_topks:
                    window_k = max(k, len(gold_chunk_id_set))
                    considered = eval_rows[:window_k]
                    covered_chunk_ids = {
                        cid
                        for rr in considered
                        for cid in rr.get("matched_chunk_ids", [])
                        if cid in gold_chunk_id_set
                    }
                    hit_map[k] = 1 if gold_chunk_id_set.issubset(covered_chunk_ids) else 0
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
                        dynamic_hit = 1 if gold_chunk_id_set.issubset(dynamic_covered_chunk_ids) else 0
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

    for summary in interaction_summary_rows:
        q = max(int(summary.get("questions", 0)), 1)
        summary["hit_at_1_rate"] = round(summary.get("hit_at_1_count", 0) / q, 4)
        summary["hit_at_5_rate"] = round(summary.get("hit_at_5_count", 0) / q, 4)
        summary["hit_at_8_rate"] = round(summary.get("hit_at_8_count", 0) / q, 4)
        summary["hit_at_dynamic_k_rate"] = round(summary.get("hit_at_dynamic_k_count", 0) / q, 4)

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
            "exit_code",
            "duration_sec",
            "questions_expected",
            "retrieval_blocks",
            "questions",
            "avg_base_top1_score",
            "avg_rerank_top1_score",
            "avg_rerank_margin_top1_top2",
            "unique_rerank_top1_chunks",
            "hit_at_1_count",
            "hit_at_1_rate",
            "hit_at_5_count",
            "hit_at_5_rate",
            "hit_at_8_count",
            "hit_at_8_rate",
            "hit_at_dynamic_k_count",
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
    plot_files = build_line_plots(
        run_dir=run_dir,
        run_summaries=interaction_summary_rows,
        question_rows=question_level_rows,
        dynamic_topk_ratio=dynamic_topk_ratio,
        dynamic_topk_ratios=dynamic_topk_ratios,
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
