"""Offline index builder for Chroma + BM25 persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config_loader import apply_cli_overrides, load_config
from core.indexer import ensure_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persisted indexes.")
    parser.add_argument("--doc-file", type=str, default="./docs/chroma/master.md")
    parser.add_argument("--url-file", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--persist-dir", type=str, default="./index_store")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "indexing.persist_dir": args.persist_dir,
    }
    config = apply_cli_overrides(config, overrides)
    persist_dir = Path(config["indexing"]["persist_dir"])

    state = ensure_index(
        doc_file=args.doc_file,
        url_file=args.url_file,
        persist_dir=persist_dir,
        embedding_model="intfloat/multilingual-e5-base",
        force_rebuild=args.force_rebuild,
        debug=args.debug,
    )

    actual_dir = Path(state["meta"]["persist_dir"])
    print(f"\nIndex ready at: {actual_dir}")
    print(f"  Chroma : {actual_dir / 'chroma'}")
    print(f"  BM25   : {actual_dir / 'bm25.pkl'}")
    print(f"  Corpus : {actual_dir / 'corpus.pkl'}")
    print(f"  Meta   : {actual_dir / 'index_meta.json'}")
    print(f"  Docs   : {state['meta'].get('doc_count')} | Chunks: {state['meta'].get('chunk_count')}")


if __name__ == "__main__":
    main()