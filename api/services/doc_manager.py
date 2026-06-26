"""Document manager: scan docs/chroma, build index from selected files, hot-reload retriever."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bm25s
from bm25s.tokenization import Tokenizer
import pickle
import json
from datetime import datetime, timezone

from core.chunker import adaptive_split_documents
from core.document_loader import _load_local_doc

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs/chroma")
MULTI_DOC_PERSIST_DIR = Path("index_store/multi_doc")
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


def _path_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(str(p.resolve()).encode("utf-8"))
    return digest.hexdigest()[:12]


def list_available_docs() -> list[dict]:
    """Scan docs/chroma for all supported files and return metadata."""
    if not DOCS_DIR.is_dir():
        logger.warning("docs/chroma directory not found")
        return []
    results = []
    for f in sorted(DOCS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "suffix": f.suffix.lower(),
            })
    return results


def _load_all_selected(selected: list[str]) -> list[Document]:
    """Load multiple doc files into a flat list of Documents."""
    all_docs: list[Document] = []
    for path_str in selected:
        p = Path(path_str)
        if not p.exists():
            logger.warning("Document not found: %s", path_str)
            continue
        try:
            all_docs.extend(_load_local_doc(p))
        except Exception as e:
            logger.error("Failed to load %s: %s", path_str, e)
    return all_docs


def build_and_persist_multi_doc_index(
    selected: list[str],
    embedding_model: str = "intfloat/multilingual-e5-base",
    debug: bool = False,
) -> dict[str, Any]:
    """Build a new index from multiple selected documents and persist to disk.

    Returns the same shape as core/indexer.py's build_and_persist_index.
    """
    selected_paths = sorted(Path(p) for p in selected)
    pdir = MULTI_DOC_PERSIST_DIR / _path_hash(*selected_paths)
    pdir.mkdir(parents=True, exist_ok=True)

    docs = _load_all_selected(selected)
    chunks = adaptive_split_documents(docs, debug=debug)

    for i, ch in enumerate(chunks):
        source = ch.metadata.get("source", "__unknown")
        ch.metadata["chunk_id"] = f"{source}::chunk_{i}"
        ch.metadata["chunk_idx"] = i

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=str(pdir / "chroma")
    )

    tokenizer = Tokenizer()
    corpus_texts = [doc.page_content for doc in chunks]
    tokens = tokenizer.tokenize(corpus_texts)
    bm25_index = bm25s.BM25()
    bm25_index.index(tokens)

    with (pdir / "bm25.pkl").open("wb") as handle:
        pickle.dump({"bm25": bm25_index, "tokenizer": tokenizer}, handle)
    with (pdir / "corpus.pkl").open("wb") as handle:
        pickle.dump(chunks, handle)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "persist_dir": str(pdir.resolve()),
        "doc_files": sorted(selected),
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_model": embedding_model,
        "path_hash": _path_hash(*selected_paths),
    }
    (pdir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if debug:
        logger.info("Multi-doc index built: %d docs, %d chunks, dir=%s", len(docs), len(chunks), pdir)

    return {
        "vectordb": vectordb,
        "bm25_index": bm25_index,
        "tokenizer": tokenizer,
        "corpus": chunks,
        "meta": meta,
    }


def load_multi_doc_index(
    selected: list[str],
    embedding_model: str = "intfloat/multilingual-e5-base",
) -> dict[str, Any] | None:
    """Load a previously built multi-doc index from disk."""
    selected_paths = sorted(Path(p) for p in selected)
    pdir = MULTI_DOC_PERSIST_DIR / _path_hash(*selected_paths)
    meta_path = pdir / "index_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=str(pdir / "chroma"), embedding_function=embeddings)
    with (pdir / "bm25.pkl").open("rb") as handle:
        bm25_payload = pickle.load(handle)
    with (pdir / "corpus.pkl").open("rb") as handle:
        corpus = pickle.load(handle)
    return {
        "vectordb": vectordb,
        "bm25_index": bm25_payload["bm25"],
        "tokenizer": bm25_payload["tokenizer"],
        "corpus": corpus,
        "meta": meta,
    }