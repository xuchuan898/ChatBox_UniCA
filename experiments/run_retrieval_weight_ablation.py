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


DEFAULT_PATH_CONFIGS = {
    "current": {"weight_vec": 1.25, "weight_bm25": 0.75},
    "equal": {"weight_vec": 1.0, "weight_bm25": 1.0},
    "vec_only": {"weight_vec": 1.0, "weight_bm25": 0.0},
    "bm25_only": {"weight_vec": 0.0, "weight_bm25": 1.0},
    "vec_heavy": {"weight_vec": 1.5, "weight_bm25": 0.5},
    "bm25_heavy": {"weight_vec": 0.75, "weight_bm25": 1.25},
}

DEFAULT_VARIANT_MODES = ["primary_only", "mapped_current", "mapped_expanded"]
DEFAULT_VARIANT_WEIGHTS = [1.0, 0.85, 0.7, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval-focused interaction ablation on one document: "
            "path weights x variant mode x secondary variant weight."
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
        default=Path("questions/questions_batch_example.txt"),
        help="Question file path.",
    )
    parser.add_argument(
        "--path-configs",
        type=str,
        nargs="*",
        default=["current", "equal", "vec_heavy", "bm25_heavy"],
        help="Path-weight presets to include in interaction runs.",
    )
    parser.add_argument(
        "--extra-path-config",
        type=str,
        nargs="*",
        default=[],
        help="Extra path config in 'name:weight_vec:weight_bm25' format.",
    )
    parser.add_argument(
        "--variant-modes",
        type=str,
        nargs="*",
        default=DEFAULT_VARIANT_MODES,
        choices=["primary_only", "mapped_current", "mapped_expanded"],
        help="Variant generation modes to include.",
    )
    parser.add_argument(
        "--variant-weights",
        type=float,
        nargs="*",
        default=DEFAULT_VARIANT_WEIGHTS,
        help="Secondary variant weights used for non-primary_only modes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k rows to keep for base/rerank output (default: 5).",
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


def parse_extra_path_config(raw: str) -> tuple[str, float, float]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --extra-path-config format: {raw!r}. Expected name:weight_vec:weight_bm25"
        )
    name = parts[0].strip()
    if not name:
        raise ValueError(f"Invalid config name in --extra-path-config: {raw!r}")
    return name, float(parts[1]), float(parts[2])


def _weight_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_run_plan(args: argparse.Namespace) -> list[dict]:
    path_configs = []
    for name in args.path_configs:
        if name not in DEFAULT_PATH_CONFIGS:
            raise ValueError(f"Unknown path config name: {name}. Available: {sorted(DEFAULT_PATH_CONFIGS)}")
        cfg = DEFAULT_PATH_CONFIGS[name].copy()
        cfg["path_config"] = name
        path_configs.append(cfg)

    for raw in args.extra_path_config:
        name, weight_vec, weight_bm25 = parse_extra_path_config(raw)
        path_configs.append(
            {"path_config": name, "weight_vec": weight_vec, "weight_bm25": weight_bm25}
        )

    variant_weights = list(dict.fromkeys(args.variant_weights))
    if not variant_weights:
        raise ValueError("--variant-weights must not be empty.")

    plan = []
    for path_cfg in path_configs:
        for variant_mode in args.variant_modes:
            weights_for_mode = [1.0] if variant_mode == "primary_only" else variant_weights
            for svw in weights_for_mode:
                run_id = (
                    f"{path_cfg['path_config']}__{variant_mode}__svw{_weight_slug(svw)}"
                )
                plan.append(
                    {
                        "run_id": run_id,
                        "path_config": path_cfg["path_config"],
                        "weight_vec": path_cfg["weight_vec"],
                        "weight_bm25": path_cfg["weight_bm25"],
                        "variant_mode": variant_mode,
                        "secondary_variant_weight": float(svw),
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


def parse_retrieval_block(block: str, top_k: int = 5) -> dict:
    question = ""
    section = ""
    base_rows = []
    rerank_rows = []

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
        if section == "base" and len(cells) >= 6:
            base_rows.append(
                {
                    "rank": int(cells[0]),
                    "score": float(cells[1]),
                    "source": cells[2],
                    "source_type": cells[3],
                    "chunk_type": cells[4],
                    "preview": cells[5],
                }
            )
        elif section == "rerank" and len(cells) >= 6:
            rerank_rows.append(
                {
                    "rank": int(cells[0]),
                    "score": float(cells[1]),
                    "source": cells[2],
                    "source_type": cells[3],
                    "chunk_type": cells[4],
                    "preview": cells[5],
                }
            )

    return {
        "question": question,
        "base_rows": base_rows[:top_k],
        "rerank_rows": rerank_rows[:top_k],
    }


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
) -> None:
    lines = [
        "# Retrieval Interaction Ablation Summary",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Doc: {doc_file}",
        f"- Question file: {question_file}",
        f"- Total runs: {len(run_summaries)}",
        f"- Questions per run: {len({r['q_id'] for r in question_rows}) if question_rows else 0}",
        "",
        "## Run Configs",
        "",
        "| Run ID | Path Config | Variant Mode | Weight Ratio (vec:bm25) | Secondary Variant Weight |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in run_summaries:
        ratio = f"{row['weight_vec']:.2f}:{row['weight_bm25']:.2f}"
        lines.append(
            f"| {row['run_id']} | {row['path_config']} | {row['variant_mode']} | "
            f"{ratio} | {row['secondary_variant_weight']:.2f} |"
        )

    lines.extend(
        [
            "",
        "## Run Summary",
        "",
        "| Run ID | Path Config | Variant Mode | Sec Variant Weight | w_vec | w_bm25 | Avg Rerank Top1 | Avg Margin(1-2) | Top1 Changed vs Baseline |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_summaries:
        lines.append(
            f"| {row['run_id']} | {row['path_config']} | {row['variant_mode']} | "
            f"{row['secondary_variant_weight']:.2f} | {row['weight_vec']:.2f} | {row['weight_bm25']:.2f} | "
            f"{row['avg_rerank_top1_score']:.4f} | {row['avg_rerank_margin_top1_top2']:.4f} | "
            f"{row['top1_changed_rate_vs_baseline']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Retrieval details in `runs.json` keep Top-{top_k} for base and rerank.",
            "- `interaction_summary.csv`: interaction-level aggregate metrics.",
            "- `question_level.csv`: per-question top retrieval signals.",
            "- `details.md`: per-question/per-run variants and rerank Top-k tables.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_markdown_details(
    run_dir: Path,
    questions: list[str],
    run_payloads: list[dict],
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
            r["path_config"],
            r["variant_mode"],
            r["secondary_variant_weight"],
        ),
    )

    for q_idx, question in enumerate(questions, start=1):
        lines.append(f"## Q{q_idx}")
        lines.append("")
        lines.append(f"Question: `{_md_escape(question)}`")
        lines.append("")

        for run in sorted_runs:
            retrieval_blocks = run.get("retrieval_blocks", [])
            variant_events = run.get("variant_events", [])
            block = retrieval_blocks[q_idx - 1] if q_idx - 1 < len(retrieval_blocks) else {}
            event = variant_events[q_idx - 1] if q_idx - 1 < len(variant_events) else {}
            variants = event.get("variants", [])
            rerank_rows = block.get("rerank_rows", [])

            lines.append(
                f"### {run['run_id']} "
                f"(path={run['path_config']}, mode={run['variant_mode']}, "
                f"svw={run['secondary_variant_weight']:.2f})"
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
        raise FileNotFoundError(f"chat_box.py not found under project root: {chat_box_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    plan = build_run_plan(args)
    questions = load_questions(question_file)
    if not questions:
        raise ValueError(f"No valid questions loaded from: {question_file}")

    ollama_info = ensure_ollama(args)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"retrieval_weight_ablation_{run_id}"
    logs_dir = run_dir / "logs"
    answers_dir = run_dir / "answers"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {project_root}", flush=True)
    print(f"[INFO] Doc file: {doc_file}", flush=True)
    print(f"[INFO] Question file: {question_file}", flush=True)
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
        cmd = [
            args.python_bin,
            str(chat_box_path),
            "--doc-file", str(doc_file),
            "--question-file", str(question_file),
            "--answer-file", str(answer_path),
            "--debug",
            "--weight-vec", str(item["weight_vec"]),
            "--weight-bm25", str(item["weight_bm25"]),
            "--variant-mode", item["variant_mode"],
            "--secondary-variant-weight", str(item["secondary_variant_weight"]),
        ]

        print(
            f"[RUN {idx}/{len(plan)}] {run_id_str} "
            f"(vec={item['weight_vec']}, bm25={item['weight_bm25']}, "
            f"mode={item['variant_mode']}, svw={item['secondary_variant_weight']})",
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
        parsed_blocks = [parse_retrieval_block(block, top_k=args.top_k) for block in blocks]
        variant_events = extract_query_variant_events(log_text)

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

            row = {
                "run_id": run_id_str,
                "path_config": item["path_config"],
                "variant_mode": item["variant_mode"],
                "secondary_variant_weight": item["secondary_variant_weight"],
                "weight_vec": item["weight_vec"],
                "weight_bm25": item["weight_bm25"],
                "q_id": q_idx,
                "lang": lang,
                "question": question,
                "variants": " || ".join(variant_event.get("variants", [])),
                "base_top1_score": base_top1["score"] if base_top1 else None,
                "base_top1_chunk_type": base_top1["chunk_type"] if base_top1 else "",
                "base_top1_preview": base_top1["preview"] if base_top1 else "",
                "rerank_top1_score": rerank_top1["score"] if rerank_top1 else None,
                "rerank_top1_chunk_type": rerank_top1["chunk_type"] if rerank_top1 else "",
                "rerank_top1_preview": rerank_top1["preview"] if rerank_top1 else "",
                "rerank_margin_top1_top2": margin,
            }
            per_run_rows.append(row)
            question_level_rows.append(row.copy())

        summary = summarize_run_rows(per_run_rows)
        summary.update(
            {
                "run_id": run_id_str,
                "path_config": item["path_config"],
                "variant_mode": item["variant_mode"],
                "secondary_variant_weight": item["secondary_variant_weight"],
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

    baseline_id = "current__mapped_current__svw0p85"
    available_ids = {row["run_id"] for row in interaction_summary_rows}
    if baseline_id not in available_ids:
        baseline_id = interaction_summary_rows[0]["run_id"] if interaction_summary_rows else ""

    baseline_rows = [r for r in question_level_rows if r["run_id"] == baseline_id]
    baseline_by_qid = {r["q_id"]: r for r in baseline_rows}
    for row in question_level_rows:
        if row["run_id"] == baseline_id:
            row["top1_changed_vs_baseline"] = 0
            continue
        base = baseline_by_qid.get(row["q_id"])
        row["top1_changed_vs_baseline"] = (
            1 if base and row["rerank_top1_preview"] != base["rerank_top1_preview"] else 0
        )

    if baseline_rows:
        for summary in interaction_summary_rows:
            if summary["run_id"] == baseline_id:
                summary["top1_changed_rate_vs_baseline"] = 0.0
                continue
            rows = [r for r in question_level_rows if r["run_id"] == summary["run_id"]]
            changed = [r["top1_changed_vs_baseline"] for r in rows]
            summary["top1_changed_rate_vs_baseline"] = round(sum(changed) / len(changed), 4) if changed else 0.0
    else:
        for summary in interaction_summary_rows:
            summary["top1_changed_rate_vs_baseline"] = 0.0

    write_json(
        run_dir / "runs.json",
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_file": str(doc_file),
            "question_file": str(question_file),
            "top_k": args.top_k,
            "baseline_run_id": baseline_id,
            "runs": run_payloads,
        },
    )

    write_csv(
        run_dir / "question_level.csv",
        question_level_rows,
        [
            "run_id",
            "path_config",
            "variant_mode",
            "secondary_variant_weight",
            "weight_vec",
            "weight_bm25",
            "q_id",
            "lang",
            "question",
            "variants",
            "base_top1_score",
            "base_top1_chunk_type",
            "base_top1_preview",
            "rerank_top1_score",
            "rerank_top1_chunk_type",
            "rerank_top1_preview",
            "rerank_margin_top1_top2",
            "top1_changed_vs_baseline",
        ],
    )

    write_csv(
        run_dir / "interaction_summary.csv",
        interaction_summary_rows,
        [
            "run_id",
            "path_config",
            "variant_mode",
            "secondary_variant_weight",
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
            "top1_changed_rate_vs_baseline",
        ],
    )

    build_markdown_summary(
        run_dir=run_dir,
        doc_file=doc_file,
        question_file=question_file,
        run_summaries=interaction_summary_rows,
        question_rows=question_level_rows,
        top_k=args.top_k,
    )
    build_markdown_details(
        run_dir=run_dir,
        questions=questions,
        run_payloads=run_payloads,
        top_k=args.top_k,
    )

    print("[INFO] Interaction ablation finished.", flush=True)
    print(f"[INFO] Baseline run: {baseline_id}", flush=True)
    print(f"[INFO] Summary: {run_dir / 'summary.md'}", flush=True)
    print(f"[INFO] Details: {run_dir / 'details.md'}", flush=True)
    print(f"[INFO] Interaction table: {run_dir / 'interaction_summary.csv'}", flush=True)
    print(f"[INFO] Question-level table: {run_dir / 'question_level.csv'}", flush=True)


if __name__ == "__main__":
    main()
