"""Offline index builder for Chroma + BM25 persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config_loader import apply_cli_overrides, load_config
from core.indexer import build_and_persist_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persisted indexes.")
    parser.add_argument("--doc-file", type=str, default="./docs/chroma/master.md")
    parser.add_argument("--url-file", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--persist-dir", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        "indexing.persist_dir": args.persist_dir,
        "indexing.force_rebuild": args.force_rebuild if args.force_rebuild else None,
    }
    config = apply_cli_overrides(config, overrides)
    persist_dir = Path(config["indexing"]["persist_dir"])
    build_and_persist_index(
        doc_file=args.doc_file,
        url_file=args.url_file,
        persist_dir=persist_dir,
        embedding_model="intfloat/multilingual-e5-base",
        debug=args.debug,
    )
    print(f"Index built at: {persist_dir}")


if __name__ == "__main__":
    main()
