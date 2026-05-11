"""Document loading utilities for local files and optional URL sources."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.documents import Document

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

try:
    import mammoth
except ImportError:
    mammoth = None


def _load_source_urls(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []
    urls = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _normalize_google_calendar_ics_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "calendar.google.com" not in parsed.netloc:
        return None
    src_values = parse_qs(parsed.query).get("src")
    if not src_values:
        return None
    calendar_id = unquote(src_values[0])
    return f"https://calendar.google.com/calendar/ical/{calendar_id}/public/basic.ics"


def _load_local_doc(doc_file: Path) -> list[Document]:
    ext = doc_file.suffix.lower()
    if ext in {".md", ".txt"}:
        docs = TextLoader(str(doc_file), encoding="utf-8").load()
        source_type = "markdown" if ext == ".md" else "txt"
        for doc in docs:
            doc.metadata["source"] = str(doc_file)
            doc.metadata["source_type"] = source_type
        return docs
    if ext == ".pdf":
        if DocumentConverter is None:
            raise ImportError("docling is required for PDF conversion.")
        result = DocumentConverter().convert(str(doc_file))
        return [Document(page_content=result.document.export_to_markdown(), metadata={"source": str(doc_file), "source_type": "pdf_md"})]
    if ext == ".docx":
        if mammoth is None:
            raise ImportError("mammoth is required for DOCX conversion.")
        with doc_file.open("rb") as handle:
            content = mammoth.convert_to_markdown(handle).value
        return [Document(page_content=content, metadata={"source": str(doc_file), "source_type": "docx_md"})]
    raise ValueError(f"Unsupported extension: {ext}")


def _load_url_doc(url: str) -> list[Document]:
    if "discord.com/channels/" in url:
        return [Document(page_content="Discord links require authenticated export.", metadata={"source": url, "source_type": "discord_link"})]
    if "calendar.google.com" in url:
        ics_url = _normalize_google_calendar_ics_url(url)
        if not ics_url:
            return []
        req = Request(ics_url, headers={"User-Agent": "MyChatBot/1.0"})
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": url, "source_type": "google_calendar_ics", "ics_url": ics_url})]
    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["source_type"] = "web"
    return docs


def load_source_documents(doc_file: str | Path, url_file: str | Path | None = None) -> list[Document]:
    """Load local document and optional URL sources."""
    docs = _load_local_doc(Path(doc_file))
    if not url_file:
        return docs
    for url in _load_source_urls(Path(url_file)):
        try:
            docs.extend(_load_url_doc(url))
        except Exception as exc:
            docs.append(Document(page_content=f"Failed loading {url}: {exc}", metadata={"source": url, "source_type": "load_error"}))
    return docs
