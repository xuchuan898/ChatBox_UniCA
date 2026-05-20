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
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .chunker import adaptive_split_documents
from .document_loader import load_source_documents


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_doc_hash(doc_file: str | Path, url_file: str | Path | None = None) -> str:
    """Compute deterministic CONTENT hash for source inputs.
    Used inside meta.json to detect content changes, NOT for folder naming.
    """
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


def _embed_cache_path(persist_dir: Path) -> Path:
    return persist_dir / "embed_cache.pkl"


def load_index_meta(persist_dir: Path) -> dict[str, Any] | None:
    path = _meta_path(persist_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_fingerprint(doc: Document) -> str:
    """Compute content hash for a single document."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _resolve_persist_dir(
    doc_file: str | Path,
    url_file: str | Path | None,
    base_persist_dir: str | Path,
) -> Path:
    """Create an isolated persist directory named after the source file PATH only.
    The directory name must be STABLE across content edits so that incremental
    updates always write into the same folder.
    """
    doc_path = Path(doc_file)
    stem = doc_path.stem
    safe_stem = "".join(c if c.isalnum() or c in ".-_" else "_" for c in stem)

    # Hash ONLY the file paths, never the file content.
    path_digest = hashlib.sha256()
    path_digest.update(str(doc_path.resolve()).encode("utf-8"))
    if url_file:
        path_digest.update(str(Path(url_file).resolve()).encode("utf-8"))
    path_hash = path_digest.hexdigest()[:8]

    return Path(base_persist_dir) / f"{safe_stem}_{path_hash}"


def _load_embed_cache(persist_dir: Path) -> dict[str, list[float]]:
    path = _embed_cache_path(persist_dir)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_embed_cache(persist_dir: Path, cache: dict[str, list[float]]) -> None:
    with _embed_cache_path(persist_dir).open("wb") as f:
        pickle.dump(cache, f)


def _diff_documents(
    old_fingerprints: dict[str, str],
    new_docs: list[Document],
) -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    """
    Compare old and new document fingerprints.
    Returns: (unchanged, modified, new, deleted, new_fingerprints)
    """
    new_fingerprints: dict[str, str] = {}
    for i, doc in enumerate(new_docs):
        source = doc.metadata.get("source", f"__inline_{i}")
        new_fingerprints[source] = _doc_fingerprint(doc)

    old_sources = set(old_fingerprints.keys())
    new_sources = set(new_fingerprints.keys())

    unchanged = [s for s in new_sources & old_sources if new_fingerprints[s] == old_fingerprints[s]]
    modified = [s for s in new_sources & old_sources if new_fingerprints[s] != old_fingerprints[s]]
    new = sorted(new_sources - old_sources)
    deleted = sorted(old_sources - new_sources)

    return unchanged, modified, new, deleted, new_fingerprints


def build_and_persist_index(
    doc_file: str | Path,
    url_file: str | Path | None,
    persist_dir: str | Path,
    embedding_model: str = "intfloat/multilingual-e5-base",
    debug: bool = False,
) -> dict[str, Any]:
    """Build vector and BM25 indexes and persist to disk (full rebuild)."""
    pdir = Path(persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    docs = load_source_documents(doc_file, url_file)
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

    with _bm25_path(pdir).open("wb") as handle:
        pickle.dump({"bm25": bm25_index, "tokenizer": tokenizer}, handle)
    with _corpus_path(pdir).open("wb") as handle:
        pickle.dump(chunks, handle)

    embed_cache: dict[str, list[float]] = {}
    if chunks:
        all_texts = [c.page_content for c in chunks]
        all_vectors = embeddings.embed_documents(all_texts)
        for txt, vec in zip(all_texts, all_vectors):
            txt_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
            embed_cache[txt_hash] = vec
        _save_embed_cache(pdir, embed_cache)
        if debug:
            print(f"[DEBUG][INDEX] embed_cache seeded with {len(embed_cache)} entries")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "persist_dir": str(pdir.resolve()),
        "doc_file": str(Path(doc_file)),
        "url_file": str(url_file) if url_file else None,
        "doc_hash": compute_doc_hash(doc_file, url_file),
        "doc_fingerprints": {
            d.metadata.get("source", f"__inline_{i}"): _doc_fingerprint(d)
            for i, d in enumerate(docs)
        },
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "embedding_model": embedding_model,
    }
    _meta_path(pdir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if debug:
        print(f"[DEBUG][INDEX] full_rebuild_done persist_dir={pdir} chunks={len(chunks)}")

    return {
        "vectordb": vectordb,
        "bm25_index": bm25_index,
        "tokenizer": tokenizer,
        "corpus": chunks,
        "meta": meta,
    }


def load_persisted_index(
    persist_dir: str | Path, embedding_model: str = "intfloat/multilingual-e5-base"
) -> dict[str, Any]:
    """Load persisted indexes."""
    pdir = Path(persist_dir)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=str(pdir / "chroma"), embedding_function=embeddings)
    with _bm25_path(pdir).open("rb") as handle:
        bm25_payload = pickle.load(handle)
    with _corpus_path(pdir).open("rb") as handle:
        corpus = pickle.load(handle)
    meta = load_index_meta(pdir) or {}

    if "persist_dir" not in meta:
        meta["persist_dir"] = str(pdir.resolve())
        _meta_path(pdir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
    """
    Load index if unchanged, else incrementally update or rebuild.

    Guarantees:
      - Each (doc_file, url_file) pair gets its own isolated directory
        named after the source file path (stable across content edits).
      - Only modified/new/deleted documents trigger re-chunking and re-embedding.
      - Unchanged document chunks are preserved as-is.
      - Embedding vectors for unchanged chunks are reused from cache.
    """
    start = datetime.now(timezone.utc)

    pdir = _resolve_persist_dir(doc_file, url_file, persist_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    if debug:
        print(f"\n{'='*60}")
        print(f"[DEBUG][INDEX] ensure_index start")
        print(f"[DEBUG][INDEX] doc_file      = {Path(doc_file).resolve()}")
        print(f"[DEBUG][INDEX] url_file      = {url_file}")
        print(f"[DEBUG][INDEX] persist_dir   = {pdir}")
        print(f"[DEBUG][INDEX] force_rebuild = {force_rebuild}")

    docs = load_source_documents(doc_file, url_file)
    if debug:
        print(f"[DEBUG][INDEX] loaded {len(docs)} source documents")

    meta = load_index_meta(pdir)
    current_doc_hash = compute_doc_hash(doc_file, url_file)

    # ------------------------------------------------------------------
    # Handle old-format meta (no doc_fingerprints but has doc_hash)
    # ------------------------------------------------------------------
    if meta and meta.get("doc_hash") == current_doc_hash and not meta.get("doc_fingerprints"):
        if debug:
            print(f"[DEBUG][INDEX] old meta format detected, doc_hash matches, auto-upgrading fingerprints")
        # Generate fingerprints from current docs and rewrite meta
        upgraded_fingerprints = {
            d.metadata.get("source", f"__inline_{i}"): _doc_fingerprint(d)
            for i, d in enumerate(docs)
        }
        meta["doc_fingerprints"] = upgraded_fingerprints
        _meta_path(pdir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if debug:
            print(f"[DEBUG][INDEX] meta upgraded with {len(upgraded_fingerprints)} fingerprints")

    old_fingerprints: dict[str, str] = meta.get("doc_fingerprints", {}) if meta else {}

    unchanged, modified, new, deleted, new_fingerprints = _diff_documents(
        old_fingerprints, docs
    )

    if debug:
        print(f"[DEBUG][INDEX] diff results:")
        print(f"  - unchanged ({len(unchanged)}): {unchanged}")
        print(f"  - modified  ({len(modified)}):  {modified}")
        print(f"  - new       ({len(new)}):       {new}")
        print(f"  - deleted   ({len(deleted)}):   {deleted}")

    # Fast path: nothing changed
    if not force_rebuild and not modified and not new and not deleted and meta:
        if debug:
            print(f"[DEBUG][INDEX] mode=load reason=all_unchanged")
        state = load_persisted_index(pdir, embedding_model)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        if debug:
            print(f"[DEBUG][INDEX] load_done seconds={elapsed:.2f} chunks={len(state['corpus'])}")
            print(f"{'='*60}")
        return state

    reason = "force_rebuild" if force_rebuild else "content_changed"
    if debug:
        print(f"[DEBUG][INDEX] mode=incremental reason={reason}")

    # ------------------------------------------------------------------
    # Load old corpus and auto-migrate if missing chunk_id
    # ------------------------------------------------------------------
    old_corpus: list[Document] = []
    if _corpus_path(pdir).exists():
        with _corpus_path(pdir).open("rb") as f:
            old_corpus = pickle.load(f)

    has_chunk_ids = all(ch.metadata.get("chunk_id") for ch in old_corpus) if old_corpus else True

    if old_corpus and not has_chunk_ids:
        if debug:
            print(f"[DEBUG][INDEX] old corpus missing chunk_id, auto-migrating {len(old_corpus)} chunks...")
        for i, ch in enumerate(old_corpus):
            source = ch.metadata.get("source", "__unknown")
            ch.metadata["chunk_id"] = f"{source}::chunk_{i}"
            ch.metadata["chunk_idx"] = i
        with _corpus_path(pdir).open("wb") as f:
            pickle.dump(old_corpus, f)
        has_chunk_ids = True
        if debug:
            print(f"[DEBUG][INDEX] corpus migration complete")

    embed_cache = _load_embed_cache(pdir)
    if debug:
        print(f"[DEBUG][INDEX] embed_cache size: {len(embed_cache)} entries")

    new_corpus: list[Document] = []
    cache_hits = 0
    cache_misses = 0

    for source in unchanged:
        old_chunks = [ch for ch in old_corpus if ch.metadata.get("source") == source]
        old_chunks.sort(key=lambda x: x.metadata.get("chunk_idx", 0))
        new_corpus.extend(old_chunks)
        if debug:
            print(f"[DEBUG][INDEX] [UNCHANGED] {source}: preserve {len(old_chunks)} chunks")

    sources_to_rechunk = set(modified + new)
    docs_to_rechunk = [
        doc
        for doc in docs
        if doc.metadata.get("source", "__unknown") in sources_to_rechunk
    ]

    fresh_chunks: list[Document] = []
    if docs_to_rechunk:
        if debug:
            srcs = [d.metadata.get("source") for d in docs_to_rechunk]
            print(f"[DEBUG][INDEX] re-chunking {len(docs_to_rechunk)} docs: {srcs}")

        fresh_chunks = adaptive_split_documents(docs_to_rechunk, debug=debug)

        for i, ch in enumerate(fresh_chunks):
            source = ch.metadata.get("source", "__unknown")
            ch.metadata["chunk_id"] = f"{source}::chunk_{i}"
            ch.metadata["chunk_idx"] = i
            ch.metadata["chunk_hash"] = hashlib.sha256(
                ch.page_content.encode("utf-8")
            ).hexdigest()

        for ch in fresh_chunks:
            ch_hash = ch.metadata["chunk_hash"]
            if ch_hash in embed_cache:
                cache_hits += 1
                ch.metadata["_embed_cache_hit"] = True
            else:
                cache_misses += 1
                ch.metadata["_embed_cache_hit"] = False

        new_corpus.extend(fresh_chunks)
        if debug:
            print(f"[DEBUG][INDEX] fresh_chunks={len(fresh_chunks)} "
                  f"embed_cache_hits={cache_hits} misses={cache_misses}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if meta and not force_rebuild and _corpus_path(pdir).exists():
        # --- Incremental mode ---
        vectordb = Chroma(
            persist_directory=str(pdir / "chroma"),
            embedding_function=embeddings,
        )

        ids_to_delete = [
            ch.metadata.get("chunk_id")
            for ch in old_corpus
            if ch.metadata.get("source") in (modified + deleted)
            and ch.metadata.get("chunk_id")
        ]
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
            if debug:
                print(f"[DEBUG][INDEX] [CHROMA] deleted {len(ids_to_delete)} stale chunks "
                      f"(modified/deleted docs)")

        if fresh_chunks:
            cached_texts: list[str] = []
            cached_metas: list[dict] = []
            cached_embeds: list[list[float]] = []
            cached_ids: list[str] = []

            texts_to_embed: list[str] = []
            metas_to_embed: list[dict] = []
            ids_to_embed: list[str] = []

            for ch in fresh_chunks:
                cid = ch.metadata["chunk_id"]
                ch_hash = ch.metadata["chunk_hash"]
                if ch.metadata.get("_embed_cache_hit"):
                    cached_texts.append(ch.page_content)
                    cached_metas.append(ch.metadata)
                    cached_embeds.append(embed_cache[ch_hash])
                    cached_ids.append(cid)
                else:
                    texts_to_embed.append(ch.page_content)
                    metas_to_embed.append(ch.metadata)
                    ids_to_embed.append(cid)

            if cached_texts:
                vectordb.add_texts(
                    texts=cached_texts,
                    metadatas=cached_metas,
                    embeddings=cached_embeds,
                    ids=cached_ids,
                )
                if debug:
                    print(f"[DEBUG][INDEX] [CHROMA] add {len(cached_texts)} chunks "
                          f"from embed_cache (skipped embedding model)")

            if texts_to_embed:
                computed_vectors = embeddings.embed_documents(texts_to_embed)
                vectordb.add_texts(
                    texts=texts_to_embed,
                    metadatas=metas_to_embed,
                    embeddings=computed_vectors,
                    ids=ids_to_embed,
                )
                for txt, vec in zip(texts_to_embed, computed_vectors):
                    txt_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                    embed_cache[txt_hash] = vec
                if debug:
                    print(f"[DEBUG][INDEX] [CHROMA] add {len(texts_to_embed)} chunks "
                          f"with fresh embeddings")

                _save_embed_cache(pdir, embed_cache)
                if debug:
                    print(f"[DEBUG][INDEX] embed_cache updated: {len(embed_cache)} total entries")
    else:
        # --- Full rebuild mode ---
        if debug:
            print(f"[DEBUG][INDEX] [CHROMA] full rebuild (from_documents)")
        vectordb = Chroma.from_documents(
            new_corpus,
            embedding=embeddings,
            persist_directory=str(pdir / "chroma"),
        )
        if new_corpus:
            all_texts = [c.page_content for c in new_corpus]
            all_vectors = embeddings.embed_documents(all_texts)
            for txt, vec in zip(all_texts, all_vectors):
                txt_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                embed_cache[txt_hash] = vec
            _save_embed_cache(pdir, embed_cache)
            if debug:
                print(f"[DEBUG][INDEX] embed_cache seeded with {len(embed_cache)} entries")

    tokenizer = Tokenizer()
    corpus_texts = [doc.page_content for doc in new_corpus]
    tokens = tokenizer.tokenize(corpus_texts)
    bm25_index = bm25s.BM25()
    bm25_index.index(tokens)

    with _bm25_path(pdir).open("wb") as handle:
        pickle.dump({"bm25": bm25_index, "tokenizer": tokenizer}, handle)
    with _corpus_path(pdir).open("wb") as handle:
        pickle.dump(new_corpus, handle)

    if debug:
        print(f"[DEBUG][INDEX] [BM25] rebuilt with {len(corpus_texts)} chunks")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "persist_dir": str(pdir.resolve()),
        "doc_file": str(Path(doc_file)),
        "url_file": str(url_file) if url_file else None,
        "doc_hash": compute_doc_hash(doc_file, url_file),
        "doc_fingerprints": new_fingerprints,
        "doc_count": len(docs),
        "chunk_count": len(new_corpus),
        "embedding_model": embedding_model,
        "unchanged_docs": unchanged,
        "modified_docs": modified,
        "new_docs": new,
        "deleted_docs": deleted,
    }
    _meta_path(pdir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    if debug:
        print(f"[DEBUG][INDEX] incremental_done seconds={elapsed:.2f}")
        print(f"[DEBUG][INDEX] final_stats: total_chunks={len(new_corpus)}, "
              f"unchanged={len(unchanged)}, modified={len(modified)}, "
              f"new={len(new)}, deleted={len(deleted)}")
        print(f"{'='*60}")

    return {
        "vectordb": vectordb,
        "bm25_index": bm25_index,
        "tokenizer": tokenizer,
        "corpus": new_corpus,
        "meta": meta,
    }