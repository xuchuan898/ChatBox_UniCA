from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from pathlib import Path
from types import ModuleType
from typing import Any


REFERENCE_SCRIPT_NAME = "run_retrieval_weight_ablation.py"
DEFAULT_INPUT_DIR = Path("experiments/results/retrieval_weight_ablation_20260428_222229")
DEFAULT_QUESTION_FILE = Path("questions/questions_batch_student_short_typo_en_fr.txt")
DEFAULT_GOLD_FILE = Path("questions/questions_batch_student_short_typo_en_fr_gold.json")
DEFAULT_DYNAMIC_TOPK_RATIO = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
DEFAULT_EVAL_TOPK = "1,5,8"
DEFAULT_TOP_K = 5


def load_reference_module() -> ModuleType:
    script_path = Path(__file__).resolve().with_name(REFERENCE_SCRIPT_NAME)
    spec = importlib.util.spec_from_file_location("run_retrieval_weight_ablation_ref", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reference script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild retrieval_weight_ablation reports from existing logs without rerunning experiments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Existing retrieval_weight_ablation result directory to read from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write rebuilt outputs to (default: same as input-dir).",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=DEFAULT_QUESTION_FILE,
        help="Question file used to determine the question order.",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=DEFAULT_GOLD_FILE,
        help="Gold file used for hit computation.",
    )
    parser.add_argument(
        "--gold-chunks-file",
        type=Path,
        default=None,
        help="Optional chunk catalog jsonl for preview-to-chunk fallback matching.",
    )
    parser.add_argument(
        "--variant-mode",
        type=str,
        default="mapped_current",
        choices=["primary_only", "mapped_current", "mapped_expanded"],
        help="Fallback variant mode if runs.json is missing.",
    )
    parser.add_argument(
        "--fixed-bm25-ratio",
        type=float,
        default=0.5,
        help="Fallback fixed bm25 ratio if runs.json is missing.",
    )
    parser.add_argument(
        "--fixed-translation-ratio",
        type=float,
        default=0.5,
        help="Fallback fixed translation ratio if runs.json is missing.",
    )
    parser.add_argument(
        "--fixed-rerank-alpha",
        type=float,
        default=0.5,
        help="Fallback fixed rerank alpha if runs.json is missing.",
    )
    parser.add_argument(
        "--eval-topk",
        type=str,
        default=DEFAULT_EVAL_TOPK,
        help="Comma-separated top-k values for strict retrieval evaluation.",
    )
    parser.add_argument(
        "--dynamic-topk-ratio",
        type=str,
        default=DEFAULT_DYNAMIC_TOPK_RATIO,
        help="Comma-separated dynamic top-k ratios in (0,1].",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many rerank rows to show in details.md.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def resolve_path(base: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (base / path)


def resolve_existing_path(project_root: Path, input_dir: Path, raw: Any | None) -> Path | None:
    if raw is None or raw == "":
        return None
    raw_text = str(raw)
    path = raw if isinstance(raw, Path) else Path(raw_text)
    candidates: list[Path] = []

    if raw_text.startswith(("/", "\\")):
        parts = [part for part in raw_text.replace("\\", "/").split("/") if part]
        if "ChatBox_UniCA" in parts:
            idx = parts.index("ChatBox_UniCA")
            suffix = Path(*parts[idx + 1 :])
            candidates.extend([project_root / suffix, input_dir / suffix])
        if parts:
            suffix = Path(*parts)
            candidates.extend([project_root / suffix, input_dir / suffix])

    candidates.extend([path] if path.is_absolute() else [project_root / path, input_dir / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_eval_topks(reference: ModuleType, raw: str) -> list[int]:
    return reference.parse_eval_topks(raw)


def parse_dynamic_topk_ratios(reference: ModuleType, raw: str) -> list[float]:
    return reference.parse_dynamic_topk_ratios(raw)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_id(run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    def _parse_float_slug(value: str) -> float:
        return float(value.replace("p", "."))

    if run_id.startswith("bm25_ratio__r"):
        m = re.match(r"^bm25_ratio__r([0-9p.]+)__tr([0-9p.]+)__ra([0-9p.]+)$", run_id)
        if not m:
            raise ValueError(f"Unrecognized run_id: {run_id}")
        bm25_ratio = _parse_float_slug(m.group(1))
        tr = _parse_float_slug(m.group(2))
        alpha = _parse_float_slug(m.group(3))
        sweep_group = "bm25_ratio"
        sweep_var = "bm25_ratio"
        sweep_value = bm25_ratio
        weight_bm25 = bm25_ratio
        weight_vec = 1.0 - bm25_ratio
        variant_mode = args.variant_mode
        secondary_variant_weight = 1.0 if variant_mode == "primary_only" else tr
    elif run_id.startswith("translation_ratio__r"):
        m = re.match(r"^translation_ratio__r([0-9p.]+)__bm25([0-9p.]+)__ra([0-9p.]+)$", run_id)
        if not m:
            raise ValueError(f"Unrecognized run_id: {run_id}")
        tr = _parse_float_slug(m.group(1))
        bm25_ratio = _parse_float_slug(m.group(2))
        alpha = _parse_float_slug(m.group(3))
        sweep_group = "translation_ratio"
        sweep_var = "translation_ratio"
        sweep_value = tr
        weight_bm25 = bm25_ratio
        weight_vec = 1.0 - bm25_ratio
        variant_mode = args.variant_mode
        secondary_variant_weight = 1.0 if variant_mode == "primary_only" else tr
    elif run_id.startswith("rerank_alpha__r"):
        m = re.match(r"^rerank_alpha__r([0-9p.]+)__bm25([0-9p.]+)__tr([0-9p.]+)$", run_id)
        if not m:
            raise ValueError(f"Unrecognized run_id: {run_id}")
        alpha = _parse_float_slug(m.group(1))
        bm25_ratio = _parse_float_slug(m.group(2))
        tr = _parse_float_slug(m.group(3))
        sweep_group = "rerank_alpha"
        sweep_var = "rerank_alpha"
        sweep_value = alpha
        weight_bm25 = bm25_ratio
        weight_vec = 1.0 - bm25_ratio
        variant_mode = args.variant_mode
        secondary_variant_weight = 1.0 if variant_mode == "primary_only" else tr
    else:
        raise ValueError(f"Unsupported run_id pattern: {run_id}")

    return {
        "run_id": run_id,
        "sweep_group": sweep_group,
        "sweep_var": sweep_var,
        "sweep_value": float(sweep_value),
        "variant_mode": variant_mode,
        "secondary_variant_weight": float(secondary_variant_weight),
        "translation_ratio": float(tr),
        "rerank_alpha": float(alpha),
        "weight_vec": float(weight_vec),
        "weight_bm25": float(weight_bm25),
    }


def discover_run_configs(input_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = input_dir / "runs.json"
    manifest: dict[str, Any] = {}
    run_items: list[dict[str, Any]] = []

    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except Exception:
            manifest = {}
        raw_runs = manifest.get("runs", [])
        if isinstance(raw_runs, list) and raw_runs:
            for item in raw_runs:
                if isinstance(item, dict) and item.get("run_id"):
                    run_items.append(item)

    if not run_items:
        logs_dir = input_dir / "logs"
        for log_path in sorted(logs_dir.glob("*.log")):
            run_items.append(parse_run_id(log_path.stem, args))

    return manifest, run_items


def row_chunk_ids(row: dict[str, Any], reference: ModuleType, chunk_catalog: dict[str, Any]) -> list[int]:
    chunk_id = row.get("chunk_id")
    if isinstance(chunk_id, int):
        return [chunk_id]
    if isinstance(chunk_id, float) and chunk_id.is_integer():
        return [int(chunk_id)]
    if isinstance(chunk_id, str) and chunk_id not in {"", "-", "unknown"}:
        try:
            return [int(chunk_id)]
        except ValueError:
            pass
    return reference.match_preview_to_chunk_ids(str(row.get("preview", "")), chunk_catalog)


def safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def rebuild_run_rows(
    reference: ModuleType,
    run_item: dict[str, Any],
    log_path: Path,
    questions: list[str],
    gold_map: dict[str, Any],
    chunk_catalog: dict[str, Any],
    eval_topks: list[int],
    dynamic_topk_ratios: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    blocks = reference.extract_retrieval_markdown_blocks(log_text)
    parsed_blocks = [reference.parse_retrieval_block(block) for block in blocks]
    variant_events = reference.extract_query_variant_events(log_text)

    max_eval_k = max(eval_topks)
    per_run_rows: list[dict[str, Any]] = []

    for q_idx, question in enumerate(questions, start=1):
        block = parsed_blocks[q_idx - 1] if q_idx - 1 < len(parsed_blocks) else {"base_rows": [], "rerank_rows": []}
        variant_event = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else {"variants": []}

        base_rows = list(block.get("base_rows", []))
        rerank_rows = list(block.get("rerank_rows", []))
        base_top1 = base_rows[0] if base_rows else None
        rerank_top1 = rerank_rows[0] if rerank_rows else None
        rerank_top2 = rerank_rows[1] if len(rerank_rows) > 1 else None

        margin = None
        if rerank_top1 and rerank_top2:
            margin = round(float(rerank_top1["score"]) - float(rerank_top2["score"]), 4)

        gold = gold_map.get(question, {})
        gold_chunk = str(gold.get("gold_chunk", ""))
        gold_chunk_ids = list(gold.get("gold_chunk_ids", []))
        gold_chunk_id_set = set(gold_chunk_ids)

        eval_limit = max(max_eval_k, len(gold_chunk_ids))
        eval_rows = rerank_rows[:eval_limit]

        hit_map = {k: 0 for k in eval_topks}
        fixed_hit_map = {k: 0 for k in range(1, 9)}
        dynamic_by_ratio = {f"{ratio:.4f}": {"k": 0, "hit": 0} for ratio in dynamic_topk_ratios}
        top1_matched_chunk_ids: list[int] = []

        if eval_rows and gold_chunk_ids:
            for rr in eval_rows:
                rr["matched_chunk_ids"] = row_chunk_ids(rr, reference, chunk_catalog)
            top1_matched_chunk_ids = list(eval_rows[0].get("matched_chunk_ids", []))

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
            "run_id": run_item["run_id"],
            "sweep_group": run_item["sweep_group"],
            "sweep_var": run_item["sweep_var"],
            "sweep_value": run_item["sweep_value"],
            "variant_mode": run_item["variant_mode"],
            "secondary_variant_weight": run_item["secondary_variant_weight"],
            "translation_ratio": run_item["translation_ratio"],
            "rerank_alpha": run_item["rerank_alpha"],
            "weight_vec": run_item["weight_vec"],
            "weight_bm25": run_item["weight_bm25"],
            "q_id": q_idx,
            "lang": reference.detect_question_language(question),
            "question": question,
            "gold_chunk_ids": "|".join(str(cid) for cid in gold_chunk_ids),
            "gold_chunk": gold_chunk,
            "variants": " || ".join(variant_event.get("variants", [])),
            "base_top1_score": base_top1["score"] if base_top1 else None,
            "base_top1_chunk_type": base_top1.get("chunk_type", "") if base_top1 else "",
            "base_top1_preview": base_top1.get("preview", "") if base_top1 else "",
            "rerank_top1_score": rerank_top1["score"] if rerank_top1 else None,
            "rerank_top1_chunk_type": rerank_top1.get("chunk_type", "") if rerank_top1 else "",
            "rerank_top1_preview": rerank_top1.get("preview", "") if rerank_top1 else "",
            "rerank_margin_top1_top2": margin,
            "top1_matched_chunk_ids": "|".join(str(cid) for cid in top1_matched_chunk_ids),
            "dynamic_k": first_dynamic.get("k", 0),
            "dynamic_ratio": first_ratio_key,
            "hit_at_dynamic_k": first_dynamic.get("hit", 0),
            "dynamic_by_ratio": json.dumps(dynamic_by_ratio, ensure_ascii=False),
        }
        # Initialize all fixed hit_at_k fields from 1 to 8
        for k in range(1, 9):
            row[f"hit_at_{k}"] = fixed_hit_map.get(k, 0)
        # Override with eval_topks values if they exist
        for k in eval_topks:
            if k <= 8:
                row[f"hit_at_{k}"] = hit_map.get(k, row.get(f"hit_at_{k}", 0))

        per_run_rows.append(row)

    summary = reference.summarize_run_rows(per_run_rows)
    summary.update(
        {
            "run_id": run_item["run_id"],
            "sweep_group": run_item["sweep_group"],
            "sweep_var": run_item["sweep_var"],
            "sweep_value": run_item["sweep_value"],
            "variant_mode": run_item["variant_mode"],
            "secondary_variant_weight": run_item["secondary_variant_weight"],
            "translation_ratio": run_item["translation_ratio"],
            "rerank_alpha": run_item["rerank_alpha"],
            "weight_vec": run_item["weight_vec"],
            "weight_bm25": run_item["weight_bm25"],
            "rerank_candidates": run_item.get("rerank_candidates"),
            "exit_code": run_item.get("exit_code"),
            "duration_sec": run_item.get("duration_sec"),
            "retrieval_blocks": len(blocks),
            "questions_expected": len(questions),
        }
    )
    return per_run_rows, summary, variant_events


def rebuild_dynamic_k_multi_ratio_plot(
    run_dir: Path,
    run_summaries: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    dynamic_topk_ratios: list[float],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if not question_rows or not dynamic_topk_ratios or len(dynamic_topk_ratios) <= 1:
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _parse_dynamic_map(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    run_id_to_rc = {}
    for summary in run_summaries:
        run_id = str(summary.get("run_id", ""))
        try:
            rc = int(summary.get("rerank_candidates", 0))
        except Exception:
            rc = 0
        if run_id:
            run_id_to_rc[run_id] = rc

    xs = sorted({int(summary.get("rerank_candidates", 0)) for summary in run_summaries if summary.get("rerank_candidates")})
    if not xs:
        return []

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

    colors_left = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_right = ["#ff7f0e", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]

    for idx, ratio in enumerate(dynamic_topk_ratios):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx][col_idx] if rows_count > 1 else axes[col_idx]
        ratio_key = f"{ratio:.4f}"

        ys_dyn: list[float] = []
        ys_hit_dyn: list[float] = []
        for rc in xs:
            rows_data = [
                r for r in question_rows
                if run_id_to_rc.get(str(r.get("run_id", "")), 0) == rc
            ]
            if not rows_data:
                ys_dyn.append(0.0)
                ys_hit_dyn.append(0.0)
                continue

            dyn_ks: list[float] = []
            dyn_hits: list[int] = []
            for row in rows_data:
                dyn_map = _parse_dynamic_map(row.get("dynamic_by_ratio", {}))
                dyn = dyn_map.get(ratio_key, {}) if isinstance(dyn_map, dict) else {}
                if not isinstance(dyn, dict):
                    dyn = {}
                dyn_ks.append(float(dyn.get("k", 0.0)))
                dyn_hits.append(int(dyn.get("hit", 0)))

            ys_dyn.append(sum(dyn_ks) / max(len(dyn_ks), 1))
            ys_hit_dyn.append(sum(dyn_hits) / max(len(dyn_hits), 1))

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
        max_dyn = max(ys_dyn) if ys_dyn else 0.0
        ax.set_ylim(0.0, max(max_dyn * 1.2, max_dyn + 0.1, 0.1))
        ax2.set_ylim(0.0, 1.05)

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8)

    for idx in range(num_ratios, rows_count * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("Dynamic-K Metrics vs Rerank Candidates (Multiple Ratios)", fontsize=14, fontweight="bold", y=0.995)
    out = plots_dir / "dynamic_k_metrics_vs_rerank_candidates_composite.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [str(out)]


def main() -> None:
    args = parse_args()
    reference = load_reference_module()

    project_root = Path(__file__).resolve().parents[1]
    # Resolve input_dir relative to project_root if it's a relative path
    input_dir = args.input_dir if args.input_dir.is_absolute() else (project_root / args.input_dir)
    output_dir = (args.output_dir.resolve() if args.output_dir else input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, run_items = discover_run_configs(input_dir, args)
    if not run_items:
        raise FileNotFoundError(f"No runs found in: {input_dir}")

    eval_topks_raw = manifest.get("eval_topks", args.eval_topk) if manifest else args.eval_topk
    dynamic_topk_ratios_raw = manifest.get("dynamic_topk_ratios", args.dynamic_topk_ratio) if manifest else args.dynamic_topk_ratio
    if isinstance(eval_topks_raw, list):
        eval_topks = sorted({int(v) for v in eval_topks_raw})
    else:
        eval_topks = parse_eval_topks(reference, str(eval_topks_raw))
    if isinstance(dynamic_topk_ratios_raw, list):
        dynamic_topk_ratios = sorted({round(float(v), 4) for v in dynamic_topk_ratios_raw})
    else:
        dynamic_topk_ratios = parse_dynamic_topk_ratios(reference, str(dynamic_topk_ratios_raw))
    top_k = int(manifest.get("top_k", args.top_k) if manifest else args.top_k)

    question_file = resolve_existing_path(project_root, input_dir, manifest.get("question_file") if manifest else args.question_file)
    gold_file = resolve_existing_path(project_root, input_dir, manifest.get("gold_file") if manifest else args.gold_file)
    gold_chunks_path = args.gold_chunks_file
    if gold_chunks_path is None and manifest:
        candidate = manifest.get("gold_chunks_file")
        if candidate:
            gold_chunks_path = Path(candidate)
    gold_chunks_file = resolve_existing_path(project_root, input_dir, gold_chunks_path)
    if gold_chunks_file is None:
        default_catalog = input_dir / "answers" / "master_md.chunks.jsonl"
        gold_chunks_file = default_catalog if default_catalog.exists() else None

    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_file}")

    questions = reference.load_questions(question_file)
    if not questions:
        raise ValueError(f"No valid questions loaded from: {question_file}")

    gold_map = reference.load_gold_map(gold_file)
    chunk_catalog = reference.load_chunk_catalog(gold_chunks_file)
    doc_file = resolve_existing_path(project_root, input_dir, manifest.get("doc_file") if manifest else None) or Path("[unknown]")

    logs_dir = input_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    run_payloads: list[dict[str, Any]] = []
    question_level_rows: list[dict[str, Any]] = []
    interaction_summary_rows: list[dict[str, Any]] = []

    run_items_by_id = {str(item["run_id"]): item for item in run_items}
    log_paths = {p.stem: p for p in sorted(logs_dir.glob("*.log"))}

    for run_item in run_items:
        run_id = str(run_item["run_id"])
        log_path = log_paths.get(run_id)
        if log_path is None:
            print(f"[WARN] Missing log file for run: {run_id}", flush=True)
            continue

        per_run_rows, summary, variant_events = rebuild_run_rows(
            reference=reference,
            run_item=run_item,
            log_path=log_path,
            questions=questions,
            gold_map=gold_map,
            chunk_catalog=chunk_catalog,
            eval_topks=eval_topks,
            dynamic_topk_ratios=dynamic_topk_ratios,
        )

        for row in per_run_rows:
            question_level_rows.append(row.copy())
        interaction_summary_rows.append(summary)

        summary["hit_at_1_rate"] = round(summary.get("hit_at_1_count", 0) / max(int(summary.get("questions", 0)), 1), 4)
        summary["hit_at_5_rate"] = round(summary.get("hit_at_5_count", 0) / max(int(summary.get("questions", 0)), 1), 4)
        summary["hit_at_8_rate"] = round(summary.get("hit_at_8_count", 0) / max(int(summary.get("questions", 0)), 1), 4)
        summary["hit_at_dynamic_k_rate"] = round(summary.get("hit_at_dynamic_k_count", 0) / max(int(summary.get("questions", 0)), 1), 4)

        run_payloads.append(
            {
                **run_item,
                "log_path": str(log_path),
                "answer_path": str(input_dir / "answers" / f"{run_id}.answers.txt"),
                "retrieval_blocks": [reference.parse_retrieval_block(block) for block in reference.extract_retrieval_markdown_blocks(log_path.read_text(encoding="utf-8", errors="ignore"))],
                "variant_events": variant_events,
                "summary": summary,
            }
        )

    if not interaction_summary_rows:
        raise RuntimeError("No runs could be rebuilt. Check the input directory and log files.")

    # Keep the original manifest information where possible, but refresh the rebuilt fields.
    rebuilt_manifest = {
        "generated_at": reference.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if hasattr(reference, "datetime") else None,
        "rebuild_from": str(input_dir),
        "doc_file": manifest.get("doc_file", ""),
        "question_file": str(question_file),
        "gold_file": str(gold_file),
        "gold_chunks_file": str(gold_chunks_file) if gold_chunks_file else "",
        "eval_topks": eval_topks,
        "dynamic_topk_ratios": dynamic_topk_ratios,
        "dynamic_topk_ratio": dynamic_topk_ratios[0],
        "bm25_ratios": manifest.get("bm25_ratios", []),
        "translation_ratios": manifest.get("translation_ratios", []),
        "rerank_alphas": manifest.get("rerank_alphas", []),
        "fixed_bm25_ratio": manifest.get("fixed_bm25_ratio", args.fixed_bm25_ratio),
        "fixed_translation_ratio": manifest.get("fixed_translation_ratio", args.fixed_translation_ratio),
        "fixed_rerank_alpha": manifest.get("fixed_rerank_alpha", args.fixed_rerank_alpha),
        "top_k": top_k,
        "runs": run_payloads,
    }

    reference.write_json(output_dir / "runs.json", rebuilt_manifest)

    reference.write_csv(
        output_dir / "question_level.csv",
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

    reference.write_csv(
        output_dir / "interaction_summary.csv",
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

    reference.build_markdown_summary(
        run_dir=output_dir,
        doc_file=doc_file,
        question_file=question_file,
        run_summaries=interaction_summary_rows,
        question_rows=question_level_rows,
        top_k=top_k,
        eval_topks=eval_topks,
        dynamic_topk_ratios=dynamic_topk_ratios,
    )
    reference.build_markdown_details(
        run_dir=output_dir,
        questions=questions,
        run_payloads=run_payloads,
        question_rows=question_level_rows,
        dynamic_topk_ratios=dynamic_topk_ratios,
        top_k=top_k,
    )

    plot_files: list[str] = []
    if not args.no_plots:
        plot_files = reference.build_line_plots(
            run_dir=output_dir,
            run_summaries=interaction_summary_rows,
            question_rows=question_level_rows,
            dynamic_topk_ratio=dynamic_topk_ratios[0],
            dynamic_topk_ratios=dynamic_topk_ratios,
        )
        plot_files.extend(
            rebuild_dynamic_k_multi_ratio_plot(
                run_dir=output_dir,
                run_summaries=interaction_summary_rows,
                question_rows=question_level_rows,
                dynamic_topk_ratios=dynamic_topk_ratios,
            )
        )
        plot_files = sorted(set(plot_files))

    print("[INFO] Rebuild finished.", flush=True)
    print(f"[INFO] Summary: {output_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Details: {output_dir / 'details.md'}", flush=True)
    print(f"[INFO] Interaction table: {output_dir / 'interaction_summary.csv'}", flush=True)
    print(f"[INFO] Question-level table: {output_dir / 'question_level.csv'}", flush=True)
    if args.no_plots:
        print("[INFO] Plots skipped by request.", flush=True)
    elif plot_files:
        print(f"[INFO] Plots generated: {len(plot_files)} under {output_dir / 'plots'}", flush=True)
    else:
        print("[WARN] Plots were not generated (matplotlib may be unavailable).", flush=True)


if __name__ == "__main__":
    main()






