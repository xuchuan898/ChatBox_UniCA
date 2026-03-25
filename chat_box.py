# Import required libraries
import argparse
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


URL_SOURCE_FILE = "./docs/chroma/source_urls.txt"
DEFAULT_DOC_FILE = "./docs/chroma/master.md"


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
    raise ValueError(f"Unsupported doc extension: {extension}. Supported: .md, .txt, .pdf")


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


def prepare_data(doc_file: str, url_file: str | None = None, debug: bool = False):
    start_total = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    start_load = time.perf_counter()
    docs = load_source_documents(doc_file=doc_file, url_file=url_file)
    load_time = time.perf_counter() - start_load

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    texts = []
    for doc in docs:
        splits = markdown_splitter.split_text(doc.page_content)
        for split_doc in splits:
            header_context = " ".join(
                [
                    split_doc.metadata.get("H1", ""),
                    split_doc.metadata.get("H2", ""),
                    split_doc.metadata.get("H3", ""),
                ]
            )
            metadata = {**doc.metadata, **split_doc.metadata}
            texts.append(
                Document(
                    page_content=header_context + "\n" + split_doc.page_content,
                    metadata=metadata,
                )
            )

    token_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=120)
    texts = token_splitter.split_documents(texts)

    start_embedding = time.perf_counter()
    vectordb = Chroma.from_documents(texts, embedding=embeddings)
    embedding_time = time.perf_counter() - start_embedding

    if debug:
        print(f"[DEBUG] Loaded documents: {len(docs)} in {load_time:.2f}s")
        print(f"[DEBUG] Total chunks after split: {len(texts)}")
        print(f"[DEBUG] Embedding + vector DB build time: {embedding_time:.2f}s")
        print(f"[DEBUG] prepare_data total time: {time.perf_counter() - start_total:.2f}s")

    return vectordb


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

def debug_print_retrieval(retriever, question: str, max_chars: int = 350) -> None:
    debug_docs = retriever.invoke(question)
    print(f"[DEBUG] Retrieved docs for question: {question}")
    print(f"[DEBUG] Retrieved count: {len(debug_docs)}")
    for idx, doc in enumerate(debug_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        preview = doc.page_content[:max_chars].replace("\n", " ")
        print(f"[DEBUG] #{idx} source_type={source_type} source={source}")
        print(f"[DEBUG] #{idx} preview={preview}")


def chatbox(vectordb, debug: bool = False):
    llm = ChatOllama(
        
        model="gemma3:1b",
            
        validate_model_on_init=True,
            
        temperature=0.8,
            
        num_predict=256,
        
    # other params ...
    )

    prompt = ChatPromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use the context to answer concisely. Keep the answer as concise as possible. Always say "Merci pour votre question!" at the end of the answer. Please answer the question in the langugae used by the question
        {context}

        Question: {input}
        Answer:""")

    # Run chain

    retriever = vectordb.as_retriever(
        search_type="mmr",  # Use MMR to diversify results
        search_kwargs={"k": 10, "fetch_k": 15}  # Retrieve 10 docs, consider 15 initially
    )

    if debug:
        print(f"[DEBUG] Vector DB collection count: {vectordb._collection.count()}")
        debug_print_retrieval(retriever, "Quels sont les cours en semestre 1?")

    # Build the document combination chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain (this replaces RetrievalQA)
    qa_chain = create_retrieval_chain(retriever, document_chain)

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

    qa_chain, retriever = chatbox(
        prepare_data(doc_file=args.doc_file, url_file=args.url_file, debug=args.debug),
        debug=args.debug,
    )

    if args.question_file:
        questions = load_questions_from_file(args.question_file)
        if not questions:
            raise ValueError(f"No valid questions found in {args.question_file}")

        answer_lines = []
        for idx, question in enumerate(questions, start=1):
            if args.debug:
                debug_print_retrieval(retriever, question)
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
        if args.debug:
            debug_print_retrieval(retriever, args.question)
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
        if args.debug:
            debug_print_retrieval(retriever, user_input)
        start_qa = time.perf_counter()
        result = qa_chain.invoke({"input": user_input})
        if args.debug:
            print(f"[DEBUG] QA invoke time: {time.perf_counter() - start_qa:.2f}s")
        print_result = result["answer"]
        print(f"--Question received: {user_input}")
        print(f"**Bot Answer**: {print_result} \\")
    


if __name__ == "__main__":
    main()
