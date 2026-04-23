# Import required libraries
import argparse
import os
import re
import time
 
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
from collections import defaultdict
from typing import List, Optional, Dict

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder

import bm25s
from bm25s.tokenization import Tokenizer

# For PDF to Markdown conversion
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

# For DOCX to Markdown conversion
try:
    import mammoth
except ImportError:
    mammoth = None

URL_SOURCE_FILE = "./docs/chroma/source_urls.txt"
DEFAULT_DOC_FILE = "./docs/chroma/master.md"

# Prompt template used for debug logging of the final LLM input.
PROMPT_TEMPLATE = """
Use ONLY the following context to answer the question.
The following context is sorted by relevance. The FIRST document is the most important.
If the answer is not found in the context, say "I don't know."
Do not add any information not present in the context.
Keep the answer concise.
Always say "Merci pour votre question!" at the end of the answer.
Answer in the same language as the question.

Context:
{context}

Question: {input}
Answer:
"""


def _load_source_urls(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        return []

    urls = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def _normalize_google_calendar_ics_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "calendar.google.com" not in parsed.netloc:
        return None

    query = parse_qs(parsed.query)
    src_values = query.get("src")
    if not src_values:
        return None

    calendar_id = unquote(src_values[0])
    return f"https://calendar.google.com/calendar/ical/{calendar_id}/public/basic.ics"


def _load_google_calendar_docs(url: str) -> list[Document]:
    ics_url = _normalize_google_calendar_ics_url(url)
    if not ics_url:
        return []

    request = Request(ics_url, headers={"User-Agent": "MyChatBot/1.0"})
    with urlopen(request, timeout=30) as response:
        calendar_text = response.read().decode("utf-8", errors="ignore")

    return [
        Document(
            page_content=calendar_text,
            metadata={"source": url, "source_type": "google_calendar_ics", "ics_url": ics_url},
        )
    ]


def _load_discord_docs(url: str) -> list[Document]:
    note = (
        "Discord channel links require authentication and cannot be scraped as public web text. "
        "To include Discord content in retrieval, export channel messages to a local file and load that file."
    )
    return [Document(page_content=note, metadata={"source": url, "source_type": "discord_link"})]


def _load_web_docs(url: str) -> list[Document]:
    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["source_type"] = "web"
    return docs


def _load_local_doc(doc_file: str) -> list[Document]:
    extension = os.path.splitext(doc_file)[1].lower()

    if extension == ".md" or extension == ".txt":
        return TextLoader(doc_file, encoding="utf-8").load()

    if extension == ".pdf":
        if DocumentConverter is None:
            raise ImportError("docling is required for PDF conversion. Install with: pip install docling")
        else:
            converter = DocumentConverter()
            result = converter.convert(doc_file)
            md_text = result.document.export_to_markdown()
            return [Document(
                page_content=md_text,
                metadata={"source": doc_file, "source_type": "pdf_md"}
            )]

    if extension == ".docx":
        if mammoth is None:
            raise ImportError("mammoth is required for DOCX conversion. Install with: pip install mammoth")
        else:
            with open(doc_file, "rb") as f:
                result = mammoth.convert_to_markdown(f)
                md_text = result.value
            return [Document(
                page_content=md_text,
                metadata={"source": doc_file, "source_type": "docx_md"}
            )]

    raise ValueError(f"Unsupported doc extension: {extension}. Supported: .md, .txt, .pdf, .docx")


def load_source_documents(doc_file: str, url_file: str | None = None) -> list[Document]:
    docs: list[Document] = []
    docs.extend(_load_local_doc(doc_file))

    if not url_file:
        return docs

    for url in _load_source_urls(url_file):
        try:
            if "discord.com/channels/" in url:
                docs.extend(_load_discord_docs(url))
            elif "calendar.google.com" in url:
                docs.extend(_load_google_calendar_docs(url))
            else:
                docs.extend(_load_web_docs(url))
        except Exception as exc:
            docs.append(
                Document(
                    page_content=f"Failed to load source URL {url}. Error: {exc}",
                    metadata={"source": url, "source_type": "load_error"},
                )
            )

    return docs


# ============================================================================
# Section 1: Generic TXT parser with rule-based structure detection
# ============================================================================

class TxtStructureParser:
    """
    Parses plain text files (TXT) into semantic chunks with type labels.
    Uses configurable regex rules to detect titles, lists, staff information, etc.
    """
    def __init__(self, custom_rules: Optional[Dict[str, str]] = None):
        # Default detection rules (regex pattern -> chunk_type)
        self.rules = {
            # Titles: all caps, numbered sections, common keywords
            r"^[A-Z][A-Z\s]{3,}$": "title",
            r"^[0-9]+\.\s+": "title",
            r"^(Introduction|Conclusion|Objectifs|Prérequis|Pédagogie|Débouchées|Rythme|Responsable|Semestre|Cours Programme)": "title",
            # Lists: bullet points or numbered items
            r"^[\-\*•]\s+": "list",
            r"^\d+\.\s+": "list",
            # Staff / responsible lines
            r"responsable|master|assistant|coordinateur": "staff",
            # Course detection (adjust as needed)
            r"ECTS|Lecteur|UE\s+": "course",
        }
        if custom_rules:
            self.rules.update(custom_rules)

    def parse(self, text: str) -> List[Document]:
        """Split text into paragraphs and assign chunk_type based on rules."""
        chunks = []
        # Split by double newline (standard paragraph separator)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunk_type = self._detect_type(para)
            chunks.append(Document(
                page_content=para,
                metadata={"chunk_type": chunk_type}
            ))
        # Merge consecutive list chunks into one block
        merged = self._merge_lists(chunks)
        return merged

    def _detect_type(self, text: str) -> str:
        for pattern, ctype in self.rules.items():
            if re.search(pattern, text, re.IGNORECASE):
                return ctype
        return "general"

    def _merge_lists(self, chunks: List[Document]) -> List[Document]:
        merged = []
        i = 0
        while i < len(chunks):
            if chunks[i].metadata.get("chunk_type") == "list":
                list_text = chunks[i].page_content
                j = i + 1
                while j < len(chunks) and chunks[j].metadata.get("chunk_type") == "list":
                    list_text += "\n" + chunks[j].page_content
                    j += 1
                merged.append(Document(
                    page_content=list_text,
                    metadata={"chunk_type": "list"}
                ))
                i = j
            else:
                merged.append(chunks[i])
                i += 1
        return merged


# ============================================================================
# Section 2: Enhanced adaptive splitter for all document types
# ============================================================================

def _is_table(content: str) -> bool:
    """Heuristic to detect markdown tables."""
    lines = content.split('\n')
    if len(lines) < 2:
        return False
    pipe_count = sum(1 for line in lines if '|' in line)
    return pipe_count > 2


def _adaptive_split_documents(docs: List[Document], debug: bool = False, save_chunks_file: str = None) -> List[Document]:
    final_chunks = []
    chunk_records = []

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]

    # Using strip_headers=False to keep the actual header text within the content
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Enhanced separators to prioritize sentence boundaries and avoid splitting mid-sentence
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    txt_parser = TxtStructureParser()

    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        content = doc.page_content

        # Simple regex to remove common PDF conversion noise like page numbers
        content = re.sub(r"(?i)page \d+ (of|/) \d+", "", content)

        if source_type in {"discord_link", "load_error"}:
            final_chunks.append(doc)
            continue

        # ------------------------------------------------------------
        # Case 1: TXT files - Semantic structure parsing
        # ------------------------------------------------------------
        if source_type == "txt":
            txt_chunks = txt_parser.parse(content)
            for chunk in txt_chunks:
                chunk.metadata.update(doc.metadata)
                chunk.metadata.setdefault("chunk_type", "general")
            final_chunks.extend(txt_chunks)
            continue

        # ------------------------------------------------------------
        # Case 2: Markdown / Converted PDF & DOCX - Hierarchical Breadcrumbs
        # ------------------------------------------------------------
        if source_type in {"web", "markdown", "pdf_md", "docx_md"} or content.strip().startswith("#"):
            try:
                splits = markdown_splitter.split_text(content)
            except Exception:
                splits = [Document(page_content=content, metadata=doc.metadata)]

            for split_doc in splits:
                # Hierarchical Breadcrumb Injection: Build a context path (H1 > H2 > H3)
                path_elements = [split_doc.metadata.get(h, "") for h in ["H1", "H2", "H3", "H4"]]
                full_path = " > ".join([p.strip() for p in path_elements if p])

                # Update metadata with the original source info
                split_doc.metadata.update(doc.metadata)

                # Prepend breadcrumbs to content for better retrieval grounding
                if full_path:
                    split_doc.page_content = f"[Context: {full_path}]\n{split_doc.page_content}"

                chunk_type = _guess_chunk_type(split_doc.page_content)
                split_doc.metadata["chunk_type"] = chunk_type

                if _is_table(split_doc.page_content) or "|" in split_doc.page_content:
                    final_chunks.append(split_doc)
                    continue

                # Secondary split if the header-based chunk is still too large
                if len(split_doc.page_content) > 1200:
                    sub_chunks = recursive_splitter.split_documents([split_doc])
                    for sub in sub_chunks:
                        sub.metadata["chunk_type"] = chunk_type
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(split_doc)
            continue

        # ------------------------------------------------------------
        # Case 3: Fallback - Paragraph and Sentence-aware splitting
        # ------------------------------------------------------------
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content]

        for para in paragraphs:
            para_doc = Document(page_content=para, metadata=doc.metadata.copy())
            chunk_type = _guess_chunk_type(para)
            para_doc.metadata["chunk_type"] = chunk_type

            if _is_table(para) or "|" in para:
                final_chunks.append(para_doc)
                continue

            if len(para) > 1200:
                sub_chunks = recursive_splitter.split_documents([para_doc])
                for sub in sub_chunks:
                    sub.metadata["chunk_type"] = chunk_type
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(para_doc)

    for idx, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = idx

    if debug:
        print(f"[DEBUG] Adaptive split: total chunks = {len(final_chunks)}")
        if final_chunks:
            lengths = [len(c.page_content) for c in final_chunks]
            print(
                f"[DEBUG] Chunk length stats: min={min(lengths)} max={max(lengths)} avg={sum(lengths) / len(lengths):.0f}")
            types = defaultdict(int)
            for c in final_chunks:
                typ = c.metadata.get("chunk_type", "unknown")
                types[typ] += 1
            print(f"[DEBUG] Chunk type distribution: {dict(types)}")

    # Save chunk info if requested
    if save_chunks_file:
        import json
        with open(save_chunks_file, "w", encoding="utf-8") as f:
            for idx, c in enumerate(final_chunks):
                record = {
                    "chunk_id": idx,
                    "content": c.page_content,
                    "chunk_type": c.metadata.get("chunk_type", "unknown"),
                    "source": c.metadata.get("source", "unknown"),
                    "length": len(c.page_content),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return final_chunks


def _guess_chunk_type(text: str) -> str:
    """
    Simple heuristic to guess the chunk type for non-TXT documents.
    Used as fallback.
    """
    text_lower = text.lower()
    if re.search(r"responsable|master|assistant|coordinateur", text_lower):
        return "staff"
    if re.search(r"ects|lecteur|ue\s+", text_lower):
        return "course"
    if re.search(r"^[\-\*•]\s+", text_lower) or re.search(r"^\d+\.\s+", text_lower):
        return "list"
    return "general"


def prepare_data(doc_file: str, url_file: str | None = None, debug: bool = False, save_chunks_file: str | None = None):
    start_total = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    start_load = time.perf_counter()
    docs = load_source_documents(doc_file=doc_file, url_file=url_file)
    load_time = time.perf_counter() - start_load

    # Adaptive document chunking (now with TXT intelligence and converted PDF/DOCX)
    texts = _adaptive_split_documents(docs, debug=debug, save_chunks_file=save_chunks_file)

    # Build vector database
    start_embedding = time.perf_counter()
    vectordb = Chroma.from_documents(texts, embedding=embeddings)
    embedding_time = time.perf_counter() - start_embedding

    # Build BM25 index for hybrid search
    corpus_texts = [doc.page_content for doc in texts]
    tokenizer = Tokenizer()
    corpus_tokens = tokenizer.tokenize(corpus_texts)
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens)

    if debug:
        print(f"[DEBUG] Loaded documents: {len(docs)} in {load_time:.2f}s")
        print(f"[DEBUG] Total chunks after split: {len(texts)}")
        print(f"[DEBUG] Embedding + vector DB build time: {embedding_time:.2f}s")
        print(f"[DEBUG] prepare_data total time: {time.perf_counter() - start_total:.2f}s")

    # Return full chunk Documents as corpus so hybrid retrieval can keep metadata.
    return vectordb, bm25_index, tokenizer, texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbox script.")
    parser.add_argument(
        "--doc-file",
        type=str,
        default=DEFAULT_DOC_FILE,
        help="Path to the local text/markdown file used for embedding.",
    )
    parser.add_argument(
        "--url-file",
        type=str,
        default=None,
        help="Optional path to a URL list file (one URL per line). If omitted, no URLs are loaded.",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Optional one-shot question. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="Optional path to a text file with batch questions, one per line",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="Optional output file path used with --question-file to save answers.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging: timing, chunk stats, and retrieval details.",
    )
    parser.add_argument(
        "--save-chunks-file",
        type=str,
        default=None,
        help="If set, save all generated chunks to this file (JSONL)",
    )
    parser.add_argument(
        "--weight-vec",
        type=float,
        default=1.25,
        help="Vector retrieval path weight used in weighted RRF fusion.",
    )
    parser.add_argument(
        "--weight-bm25",
        type=float,
        default=0.75,
        help="BM25 retrieval path weight used in weighted RRF fusion.",
    )
    parser.add_argument(
        "--secondary-variant-weight",
        type=float,
        default=0.85,
        help="Weight multiplier for non-primary query variants.",
    )
    parser.add_argument(
        "--variant-mode",
        type=str,
        choices=["primary_only", "mapped_current", "mapped_expanded"],
        default="mapped_current",
        help="Query variant generation mode used before hybrid retrieval.",
    )
    parser.add_argument(
        "--rerank-alpha",
        type=float,
        default=0.5,
        help=(
            "Dual rerank fusion weight in [0,1]. "
            "final_score = alpha*src_query_score + (1-alpha)*translated_query_score."
        ),
    )
    return parser.parse_args()


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


def load_questions_from_file(question_file: str) -> list[str]:
    questions: list[str] = []
    with open(question_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            question = _normalize_question_line(raw_line)
            if question:
                questions.append(question)
    return questions


def _short_source(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    return os.path.basename(source) if source else "unknown"


def _compact_preview(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'")


def _print_retrieval_markdown(question: str, base_scored: list[tuple[Document, float]], reranked_scored: list[tuple[Document, float]]) -> None:
    print("[DEBUG][RETRIEVE_MD_BEGIN]")
    print("### Retrieval Trace")
    print(f"- Question: `{_md_escape(question)}`")
    print(f"- Base candidates: {len(base_scored)}")
    print(f"- Reranked kept: {len(reranked_scored)}")
    print("")

    print(f"#### Base Retrieval (All {len(base_scored)})")
    print("| Rank | Chunk ID | Score | Source | Type | Chunk Type | Preview |")
    print("| ---: | ---: | ---: | --- | --- | --- | --- |")
    for idx, (doc, score) in enumerate(base_scored, 1):
        chunk_id = _md_escape(str(doc.metadata.get("chunk_id", "unknown")))
        source = _md_escape(_short_source(doc))
        source_type = _md_escape(str(doc.metadata.get("source_type", "unknown")))
        chunk_type = _md_escape(str(doc.metadata.get("chunk_type", "unknown")))
        preview = _md_escape(_compact_preview(doc.page_content))
        print(f"| {idx} | {chunk_id} | {score:.4f} | {source} | {source_type} | {chunk_type} | {preview} |")

    print("")
    print(f"#### Rerank Result (All {len(reranked_scored)})")
    print("| Rank | Chunk ID | Rerank Score | Source | Type | Chunk Type | Preview |")
    print("| ---: | ---: | ---: | --- | --- | --- | --- |")
    for idx, (doc, score) in enumerate(reranked_scored, 1):
        chunk_id = _md_escape(str(doc.metadata.get("chunk_id", "unknown")))
        source = _md_escape(_short_source(doc))
        source_type = _md_escape(str(doc.metadata.get("source_type", "unknown")))
        chunk_type = _md_escape(str(doc.metadata.get("chunk_type", "unknown")))
        preview = _md_escape(_compact_preview(doc.page_content))
        print(f"| {idx} | {chunk_id} | {score:.4f} | {source} | {source_type} | {chunk_type} | {preview} |")

    print("[DEBUG][RETRIEVE_MD_END]")


def chatbox(
    vectordb,
    bm25_index,
    tokenizer,
    corpus,
    debug: bool = False,
    return_prompt: bool = False,
    weight_vec: float = 1.25,
    weight_bm25: float = 0.75,
    secondary_variant_weight: float = 0.85,
    variant_mode: str = "mapped_current",
    rerank_alpha: float = 0.5,
):
    # Deterministic LLM settings
    llm = ChatOllama(
        model="gemma3:1b",
        temperature=0.0,
        seed=42,
        validate_model_on_init=True,
        num_predict=256,
    )

    # Optimized prompt: use only context, no hallucinations
    prompt = ChatPromptTemplate.from_template("""
        Use ONLY the following context to answer the question.
        The following context is sorted by relevance. The FIRST document is the most important.
        If the answer is not found in the context, say "I don't know."
        Do not add any information not present in the context.
        Keep the answer concise.
        Always say "Merci pour votre question!" at the end of the answer.
        Answer in the same language as the question.

        Context:
        {context}

        Question: {input}
        Answer:
    """)

    # Base retriever: MMR with larger candidate pool.
    # Keep all chunk types and apply only soft penalties later.
    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 30,
            "fetch_k": 50,
            "lambda_mult": 0.7,
        }
    )

    # Load reranker model (supports English and French)
    reranker = CrossEncoder('nvidia/llama-nemotron-rerank-1b-v2', trust_remote_code=True)
    # Reuse retrieval results for the immediate second call with the same query
    # (prompt preview call -> qa_chain internal call) to keep traces and prompt context aligned.
    retrieval_cache: dict[str, list[Document]] = {}
    corpus_docs = [
        item if isinstance(item, Document) else Document(page_content=str(item), metadata={})
        for item in corpus
    ]

    if not 0.0 <= rerank_alpha <= 1.0:
        raise ValueError(f"rerank_alpha must be in [0, 1], got: {rerank_alpha}")

    en_to_fr_base = {
        "work-study": "alternance",
        "responsible": "responsable",
        "email": "courriel",
        "address": "adresse",
        "internship": "stage",
        "duration": "duree",
        "compulsory": "obligatoire",
        "mandatory": "obligatoire",
        "pedagogical objectives": "objectifs pedagogiques",
        "program": "parcours",
        "track": "parcours",
    }
    fr_to_en_base = {
        "alternance": "work-study",
        "responsable": "responsible",
        "courriel": "email",
        "adresse": "address",
        "stage": "internship",
        "duree": "duration",
        "durée": "duration",
        "obligatoire": "mandatory",
        "objectifs pedagogiques": "pedagogical objectives",
        "objectifs pédagogiques": "pedagogical objectives",
        "parcours": "track",
    }
    en_to_fr_expanded = {
        **en_to_fr_base,
        "semester": "semestre",
        "elective": "au choix",
        "credits": "ects",
        "lecturer": "lecteur",
        "teacher": "lecteur",
        "schedule": "rythme",
        "prerequisites": "prérequis",
        "objectives": "objectifs",
        "program": "programme",
        "intern": "stage",
    }
    fr_to_en_expanded = {
        **fr_to_en_base,
        "semestre": "semester",
        "au choix": "elective",
        "credits": "ects",
        "lecteur": "lecturer",
        "rythme": "schedule",
        "prérequis": "prerequisites",
        "prerequis": "prerequisites",
        "objectifs": "objectives",
        "programme": "program",
    }

    def _translate_query_for_rerank(query: str) -> str:
        lang = _detect_query_language(query)
        mapping = en_to_fr_expanded if lang == "en" else fr_to_en_expanded
        translated = _replace_terms(query, mapping)
        translated = " ".join(translated.split())
        return translated if translated else query

    def rerank_documents(query, documents, top_n=8, return_scores=False):
        if not documents:
            return [] if not return_scores else []

        src_pairs = [(query, doc.page_content) for doc in documents]
        src_scores = [float(score) for score in reranker.predict(src_pairs)]
        translated_query = _translate_query_for_rerank(query)

        if rerank_alpha < 1.0:
            trans_pairs = [(translated_query, doc.page_content) for doc in documents]
            trans_scores = [float(score) for score in reranker.predict(trans_pairs)]
        else:
            trans_scores = src_scores

        scored = []
        for doc, src_score, trans_score in zip(documents, src_scores, trans_scores):
            fused_score = rerank_alpha * src_score + (1.0 - rerank_alpha) * trans_score
            scored.append((doc, float(fused_score), float(src_score), float(trans_score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored[:top_n]
        if return_scores:
            return top_scored
        return [doc for doc, _, _, _ in top_scored]

    def _detect_query_language(text: str) -> str:
        lower = text.lower()
        fr_markers = [
            " le ", " la ", " les ", " des ", " du ", " un ", " une ",
            "dans", "avec", "pour", "est", "sont", "responsable", "durée", "courriel",
        ]
        en_markers = [
            " the ", " and ", " for ", " with ", " what ", " which ", " is ", " are ",
            "duration", "internship", "email", "responsible",
        ]
        fr_score = sum(marker in f" {lower} " for marker in fr_markers)
        en_score = sum(marker in f" {lower} " for marker in en_markers)
        if re.search(r"[àâçéèêëîïôûùüÿœ]", lower):
            fr_score += 2
        if fr_score >= en_score + 1:
            return "fr"
        return "en"

    def _replace_terms(text: str, mapping: dict[str, str]) -> str:
        updated = text
        for source, target in mapping.items():
            updated = re.sub(rf"\b{re.escape(source)}\b", target, updated, flags=re.IGNORECASE)
        return updated

    def _build_query_variants(question: str) -> list[str]:
        base = " ".join(question.split())
        variants = [base]
        if variant_mode == "primary_only":
            return variants

        lang = _detect_query_language(base)
        mapping = en_to_fr_base if lang == "en" else fr_to_en_base
        if variant_mode == "mapped_expanded":
            mapping = en_to_fr_expanded if lang == "en" else fr_to_en_expanded

        mapped = _replace_terms(base, mapping)
        mapped = " ".join(mapped.split())
        if mapped and mapped.lower() != base.lower():
            variants.append(mapped)
        # De-duplicate while preserving order.
        variants = list(dict.fromkeys(variants))
        return variants

    def _extract_bm25_docs(query_text: str, k: int) -> list[Document]:
        query_tokens = tokenizer.tokenize(query_text)
        bm25_results, _ = bm25_index.retrieve(query_tokens, k=k)
        indices = bm25_results[0] if getattr(bm25_results, "ndim", 1) == 2 else bm25_results
        indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        docs = []
        for idx in indices:
            if 0 <= idx < len(corpus_docs):
                docs.append(corpus_docs[idx])
        return docs

    def _weighted_rrf_fusion(ranked_lists: list[tuple[list[Document], float]], k: int = 60):
        scores = defaultdict(float)
        docs_by_content = {}
        for docs, weight in ranked_lists:
            for rank, doc in enumerate(docs, 1):
                scores[doc.page_content] += weight / (k + rank)
                docs_by_content.setdefault(doc.page_content, doc)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(docs_by_content[text], score) for text, score in sorted_items]

    def _dynamic_answer_top_k(reranked_scored: list[tuple[Document, float]]) -> int:
        if not reranked_scored:
            return 0
        base_k = min(8, len(reranked_scored))
        if len(reranked_scored) <= base_k:
            return base_k
        cutoff_score = reranked_scored[base_k - 1][1]
        tail_scores = [score for _, score in reranked_scored[base_k: min(len(reranked_scored), base_k + 6)]]
        close_count = sum(1 for score in tail_scores if (cutoff_score - score) <= 0.03)
        return min(len(reranked_scored), base_k + close_count)

    def _score_stats_line(scores: list[float]) -> str:
        if not scores:
            return "count=0"
        return (
            f"count={len(scores)} min={min(scores):.4f} "
            f"avg={(sum(scores) / len(scores)):.4f} max={max(scores):.4f}"
        )

    def custom_retriever(question):
        if question in retrieval_cache:
            cached_docs = retrieval_cache.pop(question)
            if debug:
                print(f"[DEBUG][RETRIEVE][CACHE_HIT] q={question!r} docs={len(cached_docs)}")
            return cached_docs

        # 1. Build query variants for multilingual robustness.
        query_variants = _build_query_variants(question)
        corpus_size = len(corpus_docs)
        bm25_k = min(30, corpus_size)

        ranked_lists: list[tuple[list[Document], float]] = []
        for idx, q_variant in enumerate(query_variants):
            variant_weight = 1.0 if idx == 0 else secondary_variant_weight
            vec_docs = base_retriever.invoke(q_variant)
            bm25_docs = _extract_bm25_docs(q_variant, bm25_k)
            ranked_lists.append((vec_docs, weight_vec * variant_weight))
            ranked_lists.append((bm25_docs, weight_bm25 * variant_weight))

        # 2. Weighted RRF fusion across all retrieval routes.
        merged_scored = _weighted_rrf_fusion(ranked_lists, k=60)

        # 3. Keep top candidates before rerank.
        base_scored = merged_scored[:30] if len(merged_scored) > 30 else merged_scored
        candidates = [doc for doc, _ in base_scored]

        # 4. Rerank candidates.
        reranked_all_scored_detailed = rerank_documents(
            question,
            candidates,
            top_n=len(candidates),
            return_scores=True,
        )
        reranked_all_scored = [(doc, score) for doc, score, _, _ in reranked_all_scored_detailed]

        answer_top_k = _dynamic_answer_top_k(reranked_all_scored)
        reranked_for_answer = reranked_all_scored[:answer_top_k]

        if debug:
            translated_query = _translate_query_for_rerank(question)
            print(
                f"[DEBUG][RERANK_DUAL][QUERY] src={question!r} translated={translated_query!r} "
                f"alpha={rerank_alpha:.3f}"
            )
            print(f"[DEBUG][RETRIEVE] q={question!r} base_count={len(base_scored)} rerank_count={len(reranked_all_scored)}")
            print(
                f"[DEBUG][RETRIEVE][PIPELINE] variants={len(query_variants)} "
                f"routes={len(ranked_lists)} merged={len(merged_scored)} answer_top_k={answer_top_k} "
                f"weight_vec={weight_vec:.3f} weight_bm25={weight_bm25:.3f} "
                f"secondary_variant_weight={secondary_variant_weight:.3f} "
                f"variant_mode={variant_mode} rerank_alpha={rerank_alpha:.3f}"
            )
            for idx, q_variant in enumerate(query_variants, start=1):
                print(f"[DEBUG][RETRIEVE][QUERY_VARIANT] {idx}={q_variant!r}")

            base_scores = [score for _, score in base_scored]
            rerank_scores = [score for _, score in reranked_all_scored]
            print(f"[DEBUG][RETRIEVE][BASE_STATS] {_score_stats_line(base_scores)}")
            print(f"[DEBUG][RETRIEVE][RERANK_STATS] {_score_stats_line(rerank_scores)}")

            for idx, (doc, score) in enumerate(base_scored, 1):
                source = _short_source(doc)
                print(f"[DEBUG][RETRIEVE][BASE_TOP] rank={idx} score={score:.4f} source={source}")

            for idx, (doc, score) in enumerate(reranked_all_scored, 1):
                source = _short_source(doc)
                print(f"[DEBUG][RETRIEVE][RERANK_TOP] rank={idx} score={score:.4f} source={source}")
            for idx, (_, _, src_score, trans_score) in enumerate(reranked_all_scored_detailed, 1):
                print(
                    f"[DEBUG][RETRIEVE][RERANK_TOP_COMPONENT] rank={idx} "
                    f"src_score={src_score:.4f} trans_score={trans_score:.4f}"
                )

            _print_retrieval_markdown(question, base_scored, reranked_all_scored)

        docs_for_answer = [doc for doc, _ in reranked_for_answer]
        retrieval_cache[question] = docs_for_answer
        return docs_for_answer

    class CustomRetriever(BaseRetriever):
        def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return custom_retriever(query)

        async def _aget_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    retriever = CustomRetriever()

    if debug:
        print(f"[DEBUG] Vector DB collection count: {vectordb._collection.count()}")

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    if return_prompt:
        return qa_chain, retriever, prompt
    return qa_chain, retriever


def main() -> None:
    os.environ["USER_AGENT"] = "MyChatBot/1.0"
    args = parse_args()

    if not os.path.exists(args.doc_file):
        raise FileNotFoundError(f"Document file not found: {args.doc_file}")
    if args.url_file and not os.path.exists(args.url_file):
        raise FileNotFoundError(f"URL source file not found: {args.url_file}")
    if args.question_file and not os.path.exists(args.question_file):
        raise FileNotFoundError(f"Question file not found: {args.question_file}")

    vectordb, bm25_index, tokenizer, corpus = prepare_data(
        doc_file=args.doc_file,
        url_file=args.url_file,
        debug=args.debug,
        save_chunks_file=args.save_chunks_file,
    )

    qa_chain, retriever = chatbox(
        vectordb=vectordb,
        bm25_index=bm25_index,
        tokenizer=tokenizer,
        corpus=corpus,
        debug=args.debug,
        weight_vec=args.weight_vec,
        weight_bm25=args.weight_bm25,
        secondary_variant_weight=args.secondary_variant_weight,
        variant_mode=args.variant_mode,
        rerank_alpha=args.rerank_alpha,
    )

    if args.question_file:
        questions = load_questions_from_file(args.question_file)
        if not questions:
            raise ValueError(f"No valid questions found in {args.question_file}")

        answer_lines = []
        for idx, question in enumerate(questions, start=1):
            # Build prompt context from top retrieved docs (for debug logging)
            try:
                docs_for_prompt = retriever.get_relevant_documents(question)
            except Exception:
                try:
                    docs_for_prompt = retriever._get_relevant_documents(question, run_manager=None)
                except Exception:
                    docs_for_prompt = []

            context = "\n\n---\n\n".join([d.page_content for d in docs_for_prompt[:8]])
            prompt_text = PROMPT_TEMPLATE.format(context=context, input=question)
            if args.debug:
                print("[DEBUG][PROMPT_MD_BEGIN]")
                print(f"### LLM Input Prompt (Q{idx})")
                print(f"- Question: `{_md_escape(question)}`")
                print("")
                print("```text")
                print(prompt_text)
                print("```")
                print("[DEBUG][PROMPT_MD_END]")

            start_qa = time.perf_counter()
            result = qa_chain.invoke({"input": question})
            qa_time = time.perf_counter() - start_qa
            answer = result["answer"]

            print(f"[Q{idx}] {question}")
            print(f"[A{idx}] {answer}")
            if args.debug:
                print(f"[DEBUG] QA invoke time: {qa_time:.2f}s")

            answer_lines.append(f"[Q{idx}] {question}")
            answer_lines.append(f"[A{idx}] {answer}")
            if args.debug:
                answer_lines.append(f"[DEBUG] QA invoke time: {qa_time:.2f}s")
            answer_lines.append("")

        if args.answer_file:
            with open(args.answer_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(answer_lines).rstrip() + "\n")

        return

    if args.question:
        # Single question path: build and log prompt
        try:
            docs_for_prompt = retriever.get_relevant_documents(args.question)
        except Exception:
            try:
                docs_for_prompt = retriever._get_relevant_documents(args.question, run_manager=None)
            except Exception:
                docs_for_prompt = []
        context = "\n\n---\n\n".join([d.page_content for d in docs_for_prompt[:8]])
        prompt_text = PROMPT_TEMPLATE.format(context=context, input=args.question)
        if args.debug:
            print("[DEBUG][PROMPT_MD_BEGIN]")
            print(f"### LLM Input Prompt (Q1)")
            print(f"- Question: `{_md_escape(args.question)}`")
            print("")
            print("```text")
            print(prompt_text)
            print("```")
            print("[DEBUG][PROMPT_MD_END]")

        start_qa = time.perf_counter()
        result = qa_chain.invoke({"input": args.question})
        if args.debug:
            print(f"[DEBUG] QA invoke time: {time.perf_counter() - start_qa:.2f}s")
        print(f"**Bot Answer**: {result['answer']}")
        return

    print("Chat started. Type 'exit' to quit.")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            docs_for_prompt = retriever.get_relevant_documents(user_input)
        except Exception:
            try:
                docs_for_prompt = retriever._get_relevant_documents(user_input, run_manager=None)
            except Exception:
                docs_for_prompt = []
        context = "\n\n---\n\n".join([d.page_content for d in docs_for_prompt[:8]])
        prompt_text = PROMPT_TEMPLATE.format(context=context, input=user_input)
        if args.debug:
            print("[DEBUG][PROMPT_MD_BEGIN]")
            print(f"### LLM Input Prompt (interactive)")
            print(f"- Question: `{_md_escape(user_input)}`")
            print("")
            print("```text")
            print(prompt_text)
            print("```")
            print("[DEBUG][PROMPT_MD_END]")

        start_qa = time.perf_counter()
        result = qa_chain.invoke({"input": user_input})
        if args.debug:
            print(f"[DEBUG] QA invoke time: {time.perf_counter() - start_qa:.2f}s")
        print_result = result["answer"]
        print(f"--Question received: {user_input}")
        print(f"**Bot Answer**: {print_result} \\")


if __name__ == "__main__":
    main()




