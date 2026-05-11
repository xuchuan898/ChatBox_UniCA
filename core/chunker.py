"""Adaptive chunking logic for mixed-format documents."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _guess_chunk_type(text: str) -> str:
    lower = text.lower()
    if re.search(r"responsable|master|assistant|coordinateur", lower):
        return "staff"
    if re.search(r"ects|lecteur|ue\s+", lower):
        return "course"
    if re.search(r"^[\-\*•]\s+", lower) or re.search(r"^\d+\.\s+", lower):
        return "list"
    return "general"


def _is_table(content: str) -> bool:
    lines = content.splitlines()
    return len(lines) >= 2 and sum(1 for line in lines if "|" in line) > 2


def adaptive_split_documents(docs: List[Document], debug: bool = False, save_chunks_file: str | None = None) -> List[Document]:
    """Split docs with markdown-aware and recursive fallback logic."""
    final_chunks: list[Document] = []
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")], strip_headers=False)
    recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""])

    for doc in docs:
        content = re.sub(r"(?i)page \d+ (of|/) \d+", "", doc.page_content)
        source_type = doc.metadata.get("source_type", "unknown")
        if source_type in {"discord_link", "load_error"}:
            final_chunks.append(doc)
            continue
        if source_type in {"markdown", "web", "pdf_md", "docx_md"} or content.strip().startswith("#"):
            try:
                splits = md_splitter.split_text(content)
            except Exception:
                splits = [Document(page_content=content, metadata=doc.metadata.copy())]
            for split in splits:
                split.metadata.update(doc.metadata)
                path = " > ".join([split.metadata.get(h, "").strip() for h in ["H1", "H2", "H3", "H4"] if split.metadata.get(h)])
                if path:
                    split.page_content = f"[Context: {path}]\n{split.page_content}"
                split.metadata["chunk_type"] = _guess_chunk_type(split.page_content)
                if _is_table(split.page_content) or "|" in split.page_content or len(split.page_content) <= 1200:
                    final_chunks.append(split)
                else:
                    subs = recursive.split_documents([split])
                    for sub in subs:
                        sub.metadata["chunk_type"] = split.metadata["chunk_type"]
                    final_chunks.extend(subs)
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()] or [content]
        for para in paragraphs:
            para_doc = Document(page_content=para, metadata=doc.metadata.copy())
            para_doc.metadata["chunk_type"] = _guess_chunk_type(para)
            if _is_table(para) or "|" in para or len(para) <= 1200:
                final_chunks.append(para_doc)
            else:
                subs = recursive.split_documents([para_doc])
                for sub in subs:
                    sub.metadata["chunk_type"] = para_doc.metadata["chunk_type"]
                final_chunks.extend(subs)

    for idx, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = idx

    if debug:
        lengths = [len(c.page_content) for c in final_chunks] or [0]
        types = defaultdict(int)
        for c in final_chunks:
            types[c.metadata.get("chunk_type", "unknown")] += 1
        print(f"[DEBUG] Adaptive split: total={len(final_chunks)} min={min(lengths)} max={max(lengths)} avg={sum(lengths)/max(1,len(lengths)):.0f}")
        print(f"[DEBUG] Chunk type distribution: {dict(types)}")

    if save_chunks_file:
        path = Path(save_chunks_file)
        with path.open("w", encoding="utf-8") as handle:
            for chunk in final_chunks:
                handle.write(json.dumps({
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "content": chunk.page_content,
                    "chunk_type": chunk.metadata.get("chunk_type"),
                    "source": chunk.metadata.get("source"),
                    "length": len(chunk.page_content),
                }, ensure_ascii=False) + "\n")
    return final_chunks
