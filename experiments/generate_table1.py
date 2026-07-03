#!/usr/bin/env python3
"""
Generate Table 1: Retrieval Ablation for Preliminary Report
Usage: python experiments/generate_table1.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Ollama 检查与启动
# ============================================================
def ollama_is_ready(ollama_host: str = "http://127.0.0.1:11434") -> bool:
    try:
        with urlopen(f"{ollama_host.rstrip('/')}/api/tags", timeout=2):
            return True
    except (URLError, TimeoutError, OSError):
        return False


def ensure_ollama(
    ollama_bin_dir: Path,
    ollama_log_file: Path,
    ollama_host: str = "http://127.0.0.1:11434",
) -> dict:
    ollama_bin = ollama_bin_dir / "ollama"
    if sys.platform == "win32" and not ollama_bin.exists():
        ollama_bin = ollama_bin_dir / "ollama.exe"

    ollama_log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{ollama_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_NUM_GPU"] = "1"
    env["OLLAMA_HOST"] = ollama_host

    info = {"env": env, "ready": False}

    if ollama_is_ready(ollama_host):
        info["ready"] = True
        print(f"  ✅ Ollama already running at {ollama_host}")
        return info

    if not ollama_bin.exists():
        print(f"  ❌ Ollama binary not found at: {ollama_bin}")
        return info

    print(f"  🔄 Starting Ollama from: {ollama_bin}")
    with ollama_log_file.open("a", encoding="utf-8") as logf:
        subprocess.Popen(
            [str(ollama_bin), "serve"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )

    for attempt in range(15):
        time.sleep(1)
        if ollama_is_ready(ollama_host):
            info["ready"] = True
            print(f"  ✅ Ollama started (attempt {attempt + 1})")
            return info

    print(f"  ⚠️  Ollama started but not responding after 15s")
    return info


# ============================================================
# 加载 Gold 数据
# ============================================================
def load_gold_map(gold_file: Path) -> dict:
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_file}")
    payload = json.loads(gold_file.read_text(encoding="utf-8"))
    mapping = {}
    for item in payload.get("items", []):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        chunk_ids = item.get("gold_chunk_ids", [])
        mapping[question] = {
            "gold_chunk_ids": [int(cid) for cid in chunk_ids if cid is not None],
            "expected_answer": item.get("expected_answer", ""),
            "qid": item.get("qid", ""),
            "lang": item.get("lang", ""),
        }
    return mapping


def load_questions(question_file: Path) -> list[str]:
    questions = []
    with question_file.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                questions.append(q)
    return questions


# ============================================================
# 加载 Chunk Catalog（用于预览→chunk_id 映射）
# ============================================================
def load_chunk_catalog(chunk_file: Path) -> dict:
    catalog = {"chunks": []}
    if not chunk_file.exists():
        return catalog
    with chunk_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            catalog["chunks"].append({
                "chunk_id": idx,
                "content": rec.get("content", ""),
                "context": rec.get("context", ""),
            })
    return catalog


def match_preview_to_chunk_ids(preview: str, chunk_catalog: dict) -> list[int]:
    if not preview or not chunk_catalog.get("chunks"):
        return []
    norm_preview = preview.lower().replace("...", " ").strip()
    norm_preview = re.sub(r"\s+", " ", norm_preview)
    if len(norm_preview) < 10:
        return []
    matched = []
    for c in chunk_catalog["chunks"]:
        if norm_preview in c["content"].lower():
            matched.append(c["chunk_id"])
    return matched


def compute_metrics(retrieved_ids: list[int], gold_ids: list[int]) -> dict:
    if not gold_ids:
        return {"hit@1": 0, "hit@5": 0, "hit@8": 0, "mrr": 0}
    gold_set = set(gold_ids)
    results = {}
    for k in [1, 5, 8]:
        results[f"hit@{k}"] = 1 if any(g in retrieved_ids[:k] for g in gold_set) else 0
    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            mrr = 1.0 / rank
            break
    results["mrr"] = mrr
    return results


# ============================================================
# 解析检索日志
# ============================================================
def extract_retrieval_blocks(log_text: str) -> list[dict]:
    pattern = re.compile(
        r"^\[DEBUG\]\[RETRIEVE_MD_BEGIN\]\n(.*?)\n\[DEBUG\]\[RETRIEVE_MD_END\]$",
        re.DOTALL | re.MULTILINE,
    )
    blocks = []
    for match in pattern.findall(log_text):
        block_text = match.strip()
        question = ""
        rerank_rows = []
        section = ""
        for line in block_text.splitlines():
            q_match = re.match(r"^- Question:\s*`(.*)`\s*$", line.strip())
            if q_match:
                question = q_match.group(1).strip()
            if line.strip().startswith("#### Rerank Result"):
                section = "rerank"
                continue
            if line.strip().startswith("#### Base Retrieval"):
                section = "base"
                continue
            if section in ("base", "rerank") and line.strip().startswith("|") and not line.strip().startswith("| ---"):
                parts = [p.strip() for p in line.strip().split("|")]
                parts = [p for p in parts if p]
                if parts and parts[0].lower() != "rank" and len(parts) >= 6:
                    try:
                        rerank_rows.append({
                            "rank": int(parts[0]),
                            "score": float(parts[2]),
                            "source": parts[3],
                            "chunk_type": parts[4],
                            "preview": parts[5] if len(parts) > 5 else "",
                        })
                    except (ValueError, IndexError):
                        pass
        blocks.append({"question": question, "rerank_rows": rerank_rows})
    return blocks


# ============================================================
# 运行单个配置
# ============================================================
def run_config(
    config_name: str,
    args: list[str],
    project_root: Path,
    doc_file: Path,
    question_file: Path,
    gold_map: dict,
    chunk_catalog: dict,
    ollama_env: dict,
) -> dict:
    """运行一个配置，返回 metrics"""
    cmd = [
        sys.executable,
        str(project_root / "chat_box.py"),
        "--doc-file", str(doc_file),
        "--question-file", str(question_file),
        "--debug",
    ] + args

    print(f"\n  --- {config_name} ---")
    print(f"    Command: {' '.join(cmd[:6])} ...")
    print(f"    Running...", flush=True)

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=ollama_env,
    )

    print(f"    Exit code: {result.returncode}")

    if result.returncode != 0:
        if result.stderr:
            print(f"    ❌ STDERR: {result.stderr[:500]}")
        return {
            "config": config_name,
            "total_questions": 0,
            "hit@1": 0.0,
            "hit@5": 0.0,
            "hit@8": 0.0,
            "mrr": 0.0,
            "hit@1_count": 0,
            "hit@5_count": 0,
            "hit@8_count": 0,
            "error": True,
        }

    # 解析日志
    log_text = result.stdout + result.stderr
    blocks = extract_retrieval_blocks(log_text)
    print(f"    Extracted {len(blocks)} retrieval blocks")

    if not blocks:
        return {
            "config": config_name,
            "total_questions": 0,
            "hit@1": 0.0,
            "hit@5": 0.0,
            "hit@8": 0.0,
            "mrr": 0.0,
            "hit@1_count": 0,
            "hit@5_count": 0,
            "hit@8_count": 0,
            "error": True,
        }

    all_questions = load_questions(question_file)
    hits = {k: 0 for k in [1, 5, 8]}
    mrrs = []
    processed = 0

    for q_idx, question in enumerate(all_questions, start=1):
        gold = gold_map.get(question, {})
        gold_ids = gold.get("gold_chunk_ids", [])
        if not gold_ids:
            continue

        block = blocks[q_idx - 1] if q_idx - 1 < len(blocks) else {"rerank_rows": []}
        rerank_rows = block.get("rerank_rows", [])

        retrieved_ids = []
        for row in rerank_rows:
            matched = match_preview_to_chunk_ids(row.get("preview", ""), chunk_catalog)
            retrieved_ids.extend(matched)

        # 去重
        seen = set()
        unique_ids = []
        for cid in retrieved_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        metrics = compute_metrics(unique_ids[:30], gold_ids)
        for k in [1, 5, 8]:
            hits[k] += metrics[f"hit@{k}"]
        mrrs.append(metrics["mrr"])
        processed += 1

    total = len(all_questions)
    print(f"    Processed {processed}/{total} questions")

    return {
        "config": config_name,
        "total_questions": total,
        "hit@1": hits[1] / total if total else 0,
        "hit@5": hits[5] / total if total else 0,
        "hit@8": hits[8] / total if total else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
        "hit@1_count": hits[1],
        "hit@5_count": hits[5],
        "hit@8_count": hits[8],
        "error": False,
    }


# ============================================================
# Main
# ============================================================
def main():
    import os

    project_root = Path(__file__).resolve().parents[1]
    doc_file = project_root / "docs/chroma/master.md"
    question_file = project_root / "questions/questions_batch_student_short_typo_en_fr.txt"
    gold_file = project_root / "questions/questions_batch_student_short_typo_en_fr_gold.json"

    # Ollama 配置
    ollama_bin_dir = Path.home() / "ollama/bin"
    ollama_log_file = Path.home() / "ollama/ollama.log"
    ollama_host = "http://127.0.0.1:11434"

    print("=" * 60)
    print("Table 1: Retrieval Ablation")
    print("=" * 60)

    # Step 1: Ollama
    print("\n[Step 1] Checking Ollama...")
    ollama_info = ensure_ollama(
        ollama_bin_dir=ollama_bin_dir,
        ollama_log_file=ollama_log_file,
        ollama_host=ollama_host,
    )
    if not ollama_info["ready"]:
        print("  ❌ Ollama not ready. Please start it manually: ollama serve")
        return

    # Step 2: Gold data
    print("\n[Step 2] Loading gold data...")
    gold_map = load_gold_map(gold_file)
    print(f"  Loaded {len(gold_map)} gold entries")

    # Step 3: Chunk catalog
    print("\n[Step 3] Loading chunk catalog...")
    chunk_catalog = {"chunks": []}
    possible_chunk_files = [
        project_root / "experiments/results/format_compare_20260414_162147/answers/master_md.chunks.jsonl",
        project_root / "index_store/master_md/chunks.jsonl",
    ]
    for cf in possible_chunk_files:
        if cf.exists():
            chunk_catalog = load_chunk_catalog(cf)
            print(f"  Loaded {len(chunk_catalog['chunks'])} chunks from {cf}")
            break

    if not chunk_catalog["chunks"]:
        print("  ❌ No chunk catalog found. Run a warm-up first to generate it.")
        return

    # Step 4: 4 个配置（现在 --disable-rerank 已实现！）
    configs = [
        {
            "name": "Vector only (MMR)",
            "args": [
                "--weight-vec", "1.0",
                "--weight-bm25", "0.0",
                "--variant-mode", "primary_only",
                "--secondary-variant-weight", "0.0",
                "--rerank-candidates", "30",
                "--rerank-alpha", "1.0",
                "--enable-query-expansion", "false",
                "--disable-rerank", "true",      # ✅ 现在有效
            ],
        },
        {
            "name": "+ BM25 (hybrid)",
            "args": [
                "--weight-vec", "0.5",
                "--weight-bm25", "0.5",
                "--variant-mode", "primary_only",
                "--secondary-variant-weight", "0.0",
                "--rerank-candidates", "30",
                "--rerank-alpha", "1.0",
                "--enable-query-expansion", "false",
                "--disable-rerank", "true",      # ✅ 现在有效
            ],
        },
        {
            "name": "+ Cross-encoder reranker",
            "args": [
                "--weight-vec", "0.5",
                "--weight-bm25", "0.5",
                "--variant-mode", "primary_only",
                "--secondary-variant-weight", "0.0",
                "--rerank-candidates", "30",
                "--rerank-alpha", "0.5",
                "--enable-query-expansion", "false",
                "--disable-rerank", "false",
            ],
        },
        {
            "name": "+ Query expansion",
            "args": [
                "--weight-vec", "0.5",
                "--weight-bm25", "0.5",
                "--variant-mode", "mapped_current",
                "--secondary-variant-weight", "0.85",
                "--rerank-candidates", "30",
                "--rerank-alpha", "0.5",
                "--enable-query-expansion", "true",
                "--disable-rerank", "false",
            ],
        },
    ]

    print("\n[Step 4] Running 4 configurations...")
    results = []
    for cfg in configs:
        result = run_config(
            config_name=cfg["name"],
            args=cfg["args"],
            project_root=project_root,
            doc_file=doc_file,
            question_file=question_file,
            gold_map=gold_map,
            chunk_catalog=chunk_catalog,
            ollama_env=ollama_info["env"],
        )
        results.append(result)

    # Step 5: 输出 Table 1
    print("\n" + "=" * 60)
    print("Table 1: Retrieval Ablation Results")
    print("=" * 60)

    has_error = any(r.get("error", False) for r in results)
    if has_error:
        print("⚠️  Some configurations failed. See above.\n")

    print(f"{'Configuration':<30} {'Hit@1':<8} {'Hit@5':<8} {'Hit@8':<8} {'MRR':<8}")
    print("-" * 60)
    for r in results:
        status = " ❌" if r.get("error", False) else ""
        print(
            f"{r['config']:<30} "
            f"{r['hit@1']:.3f}    "
            f"{r['hit@5']:.3f}    "
            f"{r['hit@8']:.3f}    "
            f"{r['mrr']:.3f}  {status}"
        )

    print("\n" + "-" * 60)
    print(f"{'Configuration':<30} {'Hit@8 (count)':<15} {'Total':<8}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['config']:<30} "
            f"{r['hit@8_count']:>3}/{r['total_questions']:<3}      "
            f"{r['total_questions']}"
        )

    # 保存 JSON
    output_file = project_root / "experiments/results/table1_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n  ✅ Results saved to: {output_file}")


if __name__ == "__main__":
    import os
    main()