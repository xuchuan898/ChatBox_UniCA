# Import required libraries
import argparse
import os
import re
import time
import zipfile
import numpy as np
from xml.etree import ElementTree as ET
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
from collections import defaultdict
from typing import List, Tuple, Optional

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
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
    if extension in {".md", ".txt"}:
        return TextLoader(doc_file, encoding="utf-8").load()
    if extension == ".pdf":
        return PyPDFLoader(doc_file).load()
    if extension == ".docx":
        with zipfile.ZipFile(doc_file) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for para in root.findall(".//w:p", ns):
            parts = [node.text for node in para.findall(".//w:t", ns) if node.text]
            if parts:
                paragraphs.append("".join(parts))
        page_content = "\n".join(paragraphs).strip()
        return [Document(page_content=page_content, metadata={"source": doc_file, "source_type": "docx"})]
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


def _is_table(content: str) -> bool:
    """Heuristic to detect markdown tables."""
    lines = content.split('\n')
    if len(lines) < 2:
        return False
    pipe_count = sum(1 for line in lines if '|' in line)
    return pipe_count > 2


def _adaptive_split_documents(docs: List[Document], debug: bool = False) -> List[Document]:
    """
    Adaptive chunking based on document type and structure.
    - Markdown/web: header-based splitting (preserves H1-H4)
    - Plain text: paragraph-based splitting
    - Tables/ lists: kept intact
    - Recursive split if chunk too long (>1500 chars)
    """
    final_chunks = []
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        content = doc.page_content

        # Special case: discord links or error docs – keep as single chunk
        if source_type in {"discord_link", "load_error"}:
            final_chunks.append(doc)
            continue

        # Step 1: choose primary splitter based on document type
        if source_type in {"web", "markdown"} or content.strip().startswith("#"):
            # Markdown-like: use header splitter
            try:
                splits = markdown_splitter.split_text(content)
            except Exception:
                splits = [Document(page_content=content, metadata=doc.metadata)]
        else:
            # Plain text: split by paragraphs (double newline)
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [content]
            splits = [Document(page_content=p, metadata=doc.metadata.copy()) for p in paragraphs]

        # Step 2: process each split, optionally enrich with header context
        for split_doc in splits:
            # Add header context if available (for header-based splits)
            if hasattr(split_doc, 'metadata'):
                header_ctx = " ".join([
                    split_doc.metadata.get("H1", ""),
                    split_doc.metadata.get("H2", ""),
                    split_doc.metadata.get("H3", ""),
                    split_doc.metadata.get("H4", ""),
                ])
                if header_ctx:
                    split_doc.page_content = header_ctx + "\n" + split_doc.page_content

            # Check if content is a table or list – keep intact
            if _is_table(split_doc.page_content) or "|" in split_doc.page_content:
                final_chunks.append(split_doc)
                continue

            # Step 3: length control – recursive split if too long
            if len(split_doc.page_content) > 1500:
                sub_chunks = recursive_splitter.split_documents([split_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(split_doc)

    if debug:
        print(f"[DEBUG] Adaptive split: total chunks = {len(final_chunks)}")
        if final_chunks:
            lengths = [len(c.page_content) for c in final_chunks]
            print(f"[DEBUG] Chunk length stats: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.0f}")

    return final_chunks


def prepare_data(doc_file: str, url_file: str | None = None, debug: bool = False):
    start_total = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    start_load = time.perf_counter()
    docs = load_source_documents(doc_file=doc_file, url_file=url_file)
    load_time = time.perf_counter() - start_load

    # Adaptive document chunking
    texts = _adaptive_split_documents(docs, debug=debug)

    # Build vector database
    start_embedding = time.perf_counter()
    vectordb = Chroma.from_documents(texts, embedding=embeddings)
    embedding_time = time.perf_counter() - start_embedding

    # Build BM25 index for hybrid search
    corpus = [doc.page_content for doc in texts]
    tokenizer = Tokenizer()
    corpus_tokens = tokenizer.tokenize(corpus)
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens)

    if debug:
        print(f"[DEBUG] Loaded documents: {len(docs)} in {load_time:.2f}s")
        print(f"[DEBUG] Total chunks after split: {len(texts)}")
        print(f"[DEBUG] Embedding + vector DB build time: {embedding_time:.2f}s")
        print(f"[DEBUG] prepare_data total time: {time.perf_counter() - start_total:.2f}s")

    return vectordb, bm25_index, tokenizer, corpus


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
        help="Optional path to a text file containing batch questions (one per line).",
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

    print("#### Base Retrieval (Top 5)")
    print("| Rank | Score | Source | Type | Preview |")
    print("| ---: | ---: | --- | --- | --- |")
    for idx, (doc, score) in enumerate(base_scored[:5], 1):
        source = _md_escape(_short_source(doc))
        source_type = _md_escape(str(doc.metadata.get("source_type", "unknown")))
        preview = _md_escape(_compact_preview(doc.page_content))
        print(f"| {idx} | {score:.4f} | {source} | {source_type} | {preview} |")

    print("")
    print("#### Rerank Result (Top 5)")
    print("| Rank | Rerank Score | Source | Type | Preview |")
    print("| ---: | ---: | --- | --- | --- |")
    for idx, (doc, score) in enumerate(reranked_scored[:5], 1):
        source = _md_escape(_short_source(doc))
        source_type = _md_escape(str(doc.metadata.get("source_type", "unknown")))
        preview = _md_escape(_compact_preview(doc.page_content))
        print(f"| {idx} | {score:.4f} | {source} | {source_type} | {preview} |")

    print("[DEBUG][RETRIEVE_MD_END]")


def chatbox(vectordb, bm25_index, tokenizer, corpus, debug: bool = False, return_prompt: bool = False):
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

    # Base retriever: MMR with larger candidate pool
    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 50, "lambda_mult": 0.7}
    )

    # Load reranker model (supports English and French)
    reranker = CrossEncoder('nvidia/llama-nemotron-rerank-1b-v2', trust_remote_code=True)

    def rerank_documents(query, documents, top_n=8, return_scores=False):
        if not documents:
            return [] if not return_scores else []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        scored = [(doc, float(score)) for doc, score in zip(documents, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored[:top_n]
        if return_scores:
            return top_scored
        return [doc for doc, _ in top_scored]

    def rrf_fusion(vector_docs, bm25_docs, k=60):
        scores = defaultdict(float)
        for rank, doc in enumerate(vector_docs, 1):
            scores[doc.page_content] += 1 / (k + rank)
        for rank, doc in enumerate(bm25_docs, 1):
            scores[doc.page_content] += 1 / (k + rank)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(page_content=text) for text, _ in sorted_items]

    def custom_retriever(question):
        # 1. Vector retrieval (MMR)
        vec_docs = base_retriever.invoke(question)

        # 2. BM25 retrieval - dynamically adjust k
        query_tokens = tokenizer.tokenize(question)
        corpus_size = len(corpus)
        bm25_k = min(30, corpus_size)  # Don't request more than available
        bm25_results, _ = bm25_index.retrieve(query_tokens, k=bm25_k)

        # Handle both 1D and 2D result shapes
        if bm25_results.ndim == 2:
            indices = bm25_results[0]
        else:
            indices = bm25_results

        # Convert to list of ints
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        else:
            indices = list(indices)

        # Build Document list from corpus
        bm25_docs = [Document(page_content=corpus[idx]) for idx in indices]

        # 3. RRF fusion
        merged = rrf_fusion(vec_docs, bm25_docs, k=60)

        # 4. Take top 50 candidates (or fewer)
        candidates = merged[:50] if len(merged) > 50 else merged

        # 5. Rerank
        reranked_scored = rerank_documents(question, candidates, top_n=1, return_scores=True)

        if debug:
            print(f"[DEBUG][RETRIEVE] q={question!r} vec={len(vec_docs)} bm25={len(bm25_docs)} merged={len(merged)} final={len(reranked_scored)}")
            for idx, (doc, score) in enumerate(reranked_scored[:3], 1):
                source = _short_source(doc)
                print(f"[DEBUG][TOP{idx}] score={score:.4f} source={source}")

            # For backward compatibility, produce a dummy base_scored list for markdown print
            dummy_base = [(doc, 0.0) for doc in candidates[:10]]
            _print_retrieval_markdown(question, dummy_base, reranked_scored)

        return [doc for doc, _ in reranked_scored]

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
    )

    qa_chain, retriever = chatbox(
        vectordb=vectordb,
        bm25_index=bm25_index,
        tokenizer=tokenizer,
        corpus=corpus,
        debug=args.debug,
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