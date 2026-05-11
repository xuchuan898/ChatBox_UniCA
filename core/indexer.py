"""Index build/load and persistence for Chroma + BM25."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bm25s
from bm25s.tokenization import Tokenizer
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .chunker import adaptive_split_documents
from .document_loader import load_source_documents


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_doc_hash(doc_file: str | Path, url_file: str | Path | None = None) -> str:
    """Compute deterministic hash for source inputs."""
    digest = hashlib.sha256()
    doc_path = Path(doc_file)
    digest.update(str(doc_path.resolve()).encode("utf-8"))
    digest.update(_hash_file(doc_path).encode("utf-8"))
    if url_file:
        url_path = Path(url_file)
        if url_path.exists():
            digest.update(str(url_path.resolve()).encode("utf-8"))
            digest.update(_hash_file(url_path).encode("utf-8"))
    return digest.hexdigest()


def _meta_path(persist_dir: Path) -> Path:
    return persist_dir / "index_meta.json"


def _bm25_path(persist_dir: Path) -> Path:
    return persist_dir / "bm25.pkl"


def _corpus_path(persist_dir: Path) -> Path:
    return persist_dir / "corpus.pkl"


def load_index_meta(persist_dir: Path) -> dict[str, Any] | None:
    path = _meta_path(persist_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_and_persist_index(
    doc_file: str | Path,
    url_file: str | Path | None,
    persist_dir: str | Path,
    embedding_model: str = "intfloat/multilingual-e5-base",
    debug: bool = False,
) -> dict[str, Any]:
    """Build vector and BM25 indexes and persist to disk."""
    pdir = Path(persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    docs = load_source_documents(doc_file, url_file)
    chunks = adaptive_split_documents(docs, debug=debug)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=str(pdir / "chroma"))

    tokenizer = Tokenizer()
    corpus_texts = [doc.page_content for doc in chunks]
    tokens = tokenizer.tokenize(corpus_texts)
    bm25_index = bm25s.BM25()
    bm25_index.index(tokens)

    with _bm25_path(pdir).open("wb") as handle:
        pickle.dump({"bm25": bm25_index, "tokenizer": tokenizer}, handle)
    with _corpus_path(pdir).open("wb") as handle:
        pickle.dump(chunks, handle)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "doc_file": str(Path(doc_file)),
        "url_file": str(url_file) if url_file else None,
        "doc_hash": compute_doc_hash(doc_file, url_file),
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_model": embedding_model,
    }
    _meta_path(pdir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"vectordb": vectordb, "bm25_index": bm25_index, "tokenizer": tokenizer, "corpus": chunks, "meta": meta}


def load_persisted_index(persist_dir: str | Path, embedding_model: str = "intfloat/multilingual-e5-base") -> dict[str, Any]:
    """Load persisted indexes."""
    pdir = Path(persist_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=str(pdir / "chroma"), embedding_function=embeddings)
    with _bm25_path(pdir).open("rb") as handle:
        bm25_payload = pickle.load(handle)
    with _corpus_path(pdir).open("rb") as handle:
        corpus = pickle.load(handle)
    meta = load_index_meta(pdir) or {}
    return {
        "vectordb": vectordb,
        "bm25_index": bm25_payload["bm25"],
        "tokenizer": bm25_payload["tokenizer"],
        "corpus": corpus,
        "meta": meta,
    }


def ensure_index(
    doc_file: str | Path,
    url_file: str | Path | None,
    persist_dir: str | Path,
    embedding_model: str = "intfloat/multilingual-e5-base",
    force_rebuild: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Load index if hash matches, else rebuild."""
    pdir = Path(persist_dir)
    start = datetime.now(timezone.utc)
    current_hash = compute_doc_hash(doc_file, url_file)
    meta = load_index_meta(pdir)
    if not force_rebuild and meta and meta.get("doc_hash") == current_hash:
        if debug:
            print(f"[DEBUG][INDEX] mode=load reason=hash_match persist_dir={pdir}")
        state = load_persisted_index(pdir, embedding_model)
        if debug:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            print(f"[DEBUG][INDEX] load_done seconds={elapsed:.2f} chunks={state.get('meta', {}).get('chunk_count', 'unknown')}")
        return state
    reason = "force_rebuild" if force_rebuild else "hash_mismatch_or_missing_meta"
    if debug:
        print(f"[DEBUG][INDEX] mode=rebuild reason={reason} persist_dir={pdir}")
    state = build_and_persist_index(doc_file, url_file, pdir, embedding_model, debug=debug)
    if debug:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        print(f"[DEBUG][INDEX] rebuild_done seconds={elapsed:.2f} chunks={state.get('meta', {}).get('chunk_count', 'unknown')}")
    return state
