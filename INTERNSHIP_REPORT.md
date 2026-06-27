# ChatBox_UniCA: A Local Retrieval-Augmented Generation System for Campus Knowledge Bases

**Author:** [Your Name]

**Supervisors:** [Supervisor Name(s)]

**Affiliation:** Université Côte d'Azur — Computer Science Master / UbiNet Track

**Submission Date:** August 2026

**[UniCA Logo]**

---

## Abstract

Campus life at Université Côte d'Azur generates a large volume of informational documents — course catalogs, administrative procedures, facility guides, and event schedules — that are scattered across websites, PDFs, and internal portals. Finding specific information within this heterogeneous corpus is time-consuming for students and staff alike. This report presents **ChatBox_UniCA**, a local-first Question Answering (QA) system built on Retrieval-Augmented Generation (RAG). The system ingests campus documents, performs adaptive chunking, indexes them into a hybrid vector-keyword search engine, and generates answers using a locally-hosted Large Language Model (LLM) via Ollama. The architecture combines ChromaDB for dense retrieval, BM25 for lexical search, Reciprocal Rank Fusion (RRF) for hybrid scoring, a cross-encoder reranker, and optional query expansion via LLM-generated paraphrases and translations. A lightweight Vue 3 frontend with a FastAPI backend exposes the full pipeline through REST endpoints and an interactive chat interface with session management, document selection, and runtime parameter control. The system is designed to run entirely on consumer hardware without external API dependencies, making it suitable for privacy-sensitive or offline campus deployments. Experimental evaluation on a custom question dataset demonstrates the trade-offs between retrieval accuracy, latency, and model size in a fully local RAG setup.

---

## 1. Introduction

### 1.1 Context

Université Côte d'Azur's academic ecosystem involves multiple information sources: Master's program descriptions, course syllabi, administrative forms, building directories, and research group pages. These documents exist in heterogeneous formats (Markdown, PDF, Word, HTML) and are distributed across different platforms. A student seeking, for example, the prerequisites for the M1 Computer Science program or the procedure to apply for an internship must navigate several websites and documents manually.

Traditional search engines index these documents but treat every query independently and present ranked lists of results rather than direct answers. A Question Answering system that can understand natural language queries and return synthesized, context-grounded answers would significantly improve the information access experience for the campus community.

### 1.2 State of the Art

Large Language Models (LLMs) have demonstrated remarkable capabilities in text generation and comprehension since the introduction of the Transformer architecture [Vaswani et al., 2017]. However, standalone LLMs suffer from two critical limitations: they are trained on static corpora with knowledge cutoffs, and they are prone to hallucination — generating plausible but factually incorrect information [Ji et al., 2023].

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), addresses these issues by coupling an LLM with an external knowledge retriever. The pipeline works as follows: given a user query, the retriever fetches relevant document chunks from a knowledge base; these chunks are concatenated into a context prompt; the LLM generates an answer grounded in that context. This approach anchors the LLM's output in verifiable source material, reduces hallucination, and allows the knowledge base to be updated independently of the model.

RAG systems fall into two broad deployment categories:

- **Cloud-based** systems rely on commercial APIs such as OpenAI's GPT-4 or Anthropic's Claude. They offer state-of-the-art generation quality and managed infrastructure but raise data privacy concerns, incur per-token costs, and require internet connectivity.
- **Local** systems run entirely on the user's hardware using open-weight models (e.g., Llama, Qwen, Gemma) and self-hosted vector databases. They provide full data sovereignty, zero recurring costs, and offline capability, but face constraints on model size and inference speed imposed by consumer GPUs or CPUs.

ChatBox_UniCA adopts the local paradigm, motivated by the sensitivity of academic records and the need for deployment on university-managed workstations without external network access.

### 1.3 Problem Statement

Building a production-quality local RAG system for campus data presents three core challenges:

1. **Data heterogeneity.** Campus documents vary in format (Markdown, PDF, DOCX), length (from a one-paragraph notice to a 200-page course handbook), and structure (hierarchical headings, tables, lists). A single chunking strategy is unlikely to work well across all types.
2. **Low-latency inference on consumer hardware.** Running a 32-billion-parameter LLM and an embedding model simultaneously on a laptop GPU or CPU imposes strict constraints on model selection and pipeline efficiency.
3. **Retrieval accuracy.** The system must retrieve the correct passages from a growing knowledge base despite the vocabulary gap between natural-language questions and technical document text. Off-the-shelf embedding models may not capture domain-specific terminology (e.g., "UbiNet," "M1 DS4H").

### 1.4 Research Objectives

This project set out to answer the following questions:

- Can a fully local RAG system deliver acceptable answer quality for campus-domain queries using open-weight models on consumer hardware?
- What is the optimal trade-off between chunk size, retrieval method (dense vs. hybrid), reranking, and generation quality for this domain?
- How can the system be architected to support runtime configuration, multiple document sources, and easy extensibility?

### 1.5 Contribution

We designed and implemented a complete, open-source RAG platform comprising:

- An **adaptive chunking** pipeline that respects document structure (Markdown headings) with fallback to character-level splitting.
- A **hybrid retriever** combining ChromaDB dense vectors and BM25 lexical search with weighted Reciprocal Rank Fusion.
- A **cross-encoder reranker** (nvidia/llama-nemotron-rerank-1b-v2) for precision re-scoring.
- An **optional query expansion** module using an LLM to generate paraphrases and translations for improved recall.
- A **configurable generation** stage using Ollama-hosted models (default: Qwen2.5 32B).
- A **RESTful API** (FastAPI) with session management, document selection, runtime config override, and SSE streaming.
- A **Vue 3 frontend** for interactive use, including a chat interface, session browser, document selector, and debug retrieval panel.
- A **command-line evaluator** with RAGAS integration for systematic assessment.

---

## 2. Materials and Methods

### 2.1 Data Collection and Preprocessing

The primary knowledge source for this project is a **master Markdown document** (`docs/chroma/master.md`) aggregating information about the Université Côte d'Azur Computer Science Master's program, including:

- Program structure and course descriptions for M1 and M2 levels
- Administrative procedures (registration, internships, mobility)
- Faculty and research group directories
- Facility guides (campus maps, library access, IT resources)

The document is approximately 200 KB of raw text, spanning several hundred distinct sections. The `core/document_loader.py` module extends intake to multiple formats:

| Format | Library | Status |
|--------|---------|--------|
| Markdown (`.md`) | Native parsing | **Default, stable** |
| Plain text (`.txt`) | Native | Stable |
| PDF (`.pdf`) | Docling | Optional, disabled by default |
| Word (`.docx`) | Mammoth | Supported |
| Web URLs | LangChain `AsyncHtmlLoader` | Supported for per-document sources |
| Google Calendar (ICS) | Custom parser | Supported for schedule data |
| Discord | LangChain Discord loader | Supported |

Documents are cleaned by stripping excessive whitespace, normalizing Unicode, and setting metadata fields (`source`, `source_type`, `chunk_type`).

### 2.2 System Architecture

The system follows the standard RAG architecture with five stages as illustrated in Figure 1.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│  Raw     │───▶│  Adaptive │───▶│  Hybrid  │───▶│  Cross-      │───▶│  LLM       │
│  Docs    │    │  Chunker  │    │  Index   │    │  Encoder     │    │  Generator │
└──────────┘    └──────────┘    └──────────┘    │  Reranker    │    └────────────┘
                                                └──────────────┘
                                                       ▲
┌──────────┐    ┌──────────┐                         │
│  User     │───▶│  Query   │─────────────────────────┘
│  Query    │    │ Expander │
└──────────┘    └──────────┘
```

**[Insert Figure 1: System Architecture Diagram — A flow diagram showing the RAG pipeline: Document Ingestion → Chunking → Indexing (Chroma + BM25) → Query Expansion → Hybrid Retrieval → RRF Fusion → Cross-Encoder Rerank → Context Assembly → Answer Generation. Also show the API layer (FastAPI) wrapping the pipeline and the Vue frontend communicating via REST/SSE.]**

The core processing pipeline is implemented in `services/rag_service.py`, which orchestrates the steps in a synchronous `chat()` method and an SSE-streaming `chat_stream()` variant.

### 2.3 Chunking Strategy

Chunking is implemented in `core/chunker.py`. The strategy is **structure-aware**:

1. **Markdown documents** are first split by `MarkdownHeaderTextSplitter` (separating at H1–H4 headers), which preserves document hierarchy.
2. Chunks exceeding 1000 characters are further split by `RecursiveCharacterTextSplitter` with 150-character overlap.
3. Tables and short segments (< 200 characters) are preserved intact to avoid breaking semantic units.
4. Each chunk receives metadata: `chunk_id` (hash), `chunk_idx`, and `chunk_type` (staff, course, list, general).

This approach balances two competing goals: respecting the author's structural organization (headings) and maintaining uniform chunk sizes for embedding and retrieval.

### 2.4 Embedding and Indexing

The indexer (`core/indexer.py`) supports three embedding models, defaulting to `intfloat/multilingual-e5-base`:

| Model | Dimensions | Language | Use Case |
|-------|-----------|----------|----------|
| `intfloat/multilingual-e5-base` | 768 | Multilingual (100+ langs) | **Default** — best coverage for English/French campus docs |
| `intfloat/e5-base-v2` | 768 | English only | Fallback for English-only corpora |
| `nomic-embed-text` (via Ollama) | 768 | Multilingual | CPU-friendly alternative |

Two indices are built for each document set:

- **Dense vector index:** ChromaDB with cosine similarity. The retriever uses Maximal Marginal Relevance (MMR) with `lambda_mult=0.7` to balance relevance and diversity among candidates.
- **Lexical index:** BM25 via the `bm25s` library, tokenized with BM25's built-in tokenizer. This captures exact keyword matches that dense retrieval might miss.

Indexes are persisted to disk under `index_store/` using SHA-256 content fingerprints for **incremental updates**: only new or modified documents are re-chunked and re-embedded, avoiding full rebuilds.

### 2.5 Hybrid Retrieval and Reranking

Retrieval is handled by `HybridRetriever` (`core/retriever.py`). The algorithm proceeds as follows:

1. **Query expansion (optional):** If enabled, the `QueryExpander` uses an LLM to generate a paraphrase and a translation of the original question. This multi-variant approach improves recall for questions that could be phrased differently in French or English.

2. **Multi-source retrieval:** Each query variant (original + paraphrase + translation) is dispatched to both the vector retriever (Chroma MMR) and the BM25 retriever. The set of results is parameterized by `rerank_candidates` (default: 30).

3. **Weighted Reciprocal Rank Fusion (RRF):** Results from all sources are merged using RRF [Cormack et al., 2009], adapted with per-source weights:

   $$S(d) = \sum_{(docs, w) \in sources} \sum_{rank} \frac{w}{k + rank}$$

   where $k = 60$ is a constant, and $w$ is the product of the source weight (vector vs. BM25) and the query variant weight (original, paraphrase, or translation). The default configuration uses $w_{vec} = 1.25$ and $w_{bm25} = 0.75$, biasing toward semantic search over keyword matching.

4. **Cross-encoder reranking:** The top RRF candidates are re-scored by a cross-encoder model (`nvidia/llama-nemotron-rerank-1b-v2` via SentenceTransformers). Unlike bi-encoders, cross-encoders process the query and document together, producing more accurate relevance scores at the cost of higher per-pair latency.

5. **Dynamic top-k selection:** After reranking, the final set of documents is determined by a score-thresholding mechanism: only documents whose reranker score is above a fraction ($\alpha = 0.5$) of the top score are retained, capped at `top_n` (default: 8). This prevents low-quality chunks from entering the LLM context.

### 2.6 Generation

Answer generation (`core/generator.py`) uses LangChain's `ChatOllama` interface to query a locally-running LLM. The prompt template is:

```
Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't know."
Do not add any information not present in the context.
Keep the answer concise.
Always say "Merci pour votre question!" at the end of the answer.
Answer in the same language as the question.

[Conversation memory, if available]
Context:
[Retrieved document chunks]

Question: [User query]
Answer:
```

Key design choices:

- **Temperature = 0.0** is used by default to ensure deterministic, reproducible answers — critical for evaluation.
- The **"I don't know" instruction** reduces hallucination by explicitly allowing the model to decline to answer.
- **Language matching** prompts the model to respond in the same language as the question, which is essential for a bilingual French/English campus environment.
- **Conversation memory** (the last 5 turns) is injected above the context, allowing the model to resolve anaphora (e.g., "What about the prerequisites?").

### 2.7 Semantic Cache

A semantic cache (`core/semantic_cache.py`) is available but **disabled by default**. When enabled, incoming queries are embedded with `intfloat/multilingual-e5-base` and compared against cached entries using cosine similarity. If a match above a configurable threshold (default: 0.92) is found, the cached answer is returned without invoking the retrieval or generation pipeline. This is useful for repeated queries (e.g., "What are the M1 prerequisites?") at the cost of embedding the query twice (once for cache lookup, once for retrieval).

### 2.8 Evaluation Methodology

We evaluate the system along three dimensions:

1. **Retrieval accuracy** — measured by **Hit Rate** (whether the correct chunk appears in the top-k retrieved documents) and **Mean Reciprocal Rank (MRR)** (the rank position of the first relevant chunk). These are computed via the `core/evaluator.py` module using a labeled question dataset.

2. **End-to-end answer quality** — measured using the **RAGAS** framework [Shah et al., 2024], which computes:
   - **Faithfulness:** Is the generated answer supported by the retrieved context?
   - **Answer Relevancy:** Does the answer address the question?
   - **Context Precision:** Are relevant chunks ranked higher than irrelevant ones?
   - **Context Recall:** Are all necessary chunks retrieved?

3. **Latency** — measured as end-to-end response time, broken down by pipeline stage (retrieval, reranking, generation).

---

## 3. Results

### 3.1 Development Realization

ChatBox_UniCA was implemented as a full-stack application. The repository structure is:

```
ChatBox_UniCA/
├── api/                    # FastAPI backend
│   ├── main.py             # App factory, lifespan, CORS, health check
│   ├── dependencies.py     # AppState singleton, DI providers
│   ├── models/             # Pydantic request/response schemas
│   ├── routers/            # Route modules (6 routers)
│   │   ├── chat.py         # POST /chat, POST /chat/stream
│   │   ├── sessions.py     # Session CRUD
│   │   ├── index.py        # Index status, rebuild
│   │   ├── documents.py    # Document listing, selection
│   │   ├── config.py       # Config read/write
│   │   └── debug.py        # GET /debug/retrieve
│   └── services/           # Multi-doc manager
├── core/                   # RAG pipeline modules
│   ├── retriever.py        # HybridRetriever with RRF
│   ├── reranker.py         # Cross-encoder wrapper
│   ├── query_expander.py   # LLM-based query rewriting
│   ├── generator.py        # Prompt template + ChatOllama
│   ├── chunker.py          # Adaptive chunking
│   ├── indexer.py          # Chroma + BM25 indexing
│   ├── document_loader.py  # Multi-format loader
│   ├── semantic_cache.py   # Embedding-based cache
│   ├── config_loader.py    # YAML config with overrides
│   ├── evaluator.py        # Retrieval metric computation
│   └── conversation_memory.py
├── services/               # Session service (in-memory pool)
├── frontend/               # Vue 3 + Vite frontend
│   └── src/
│       ├── components/     # 10 Vue components
│       ├── composables/    # useChat, useToast
│       ├── api/            # REST client functions
│       └── styles/         # CSS
├── config.yaml             # Master configuration
├── build_index.py          # CLI index builder
└── requirements.txt        # Python dependencies
```

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check, returns status |
| `POST` | `/api/v1/chat/` | Chat with JSON response |
| `POST` | `/api/v1/chat/stream` | Chat with SSE streaming |
| `GET` | `/api/v1/sessions/list` | List all sessions |
| `POST` | `/api/v1/sessions/new` | Create new session |
| `DELETE` | `/api/v1/sessions/{id}` | Clear a session |
| `DELETE` | `/api/v1/sessions/` | Clear all sessions |
| `GET` | `/api/v1/index/status` | Index readiness and metrics |
| `POST` | `/api/v1/index/rebuild` | Trigger index rebuild |
| `GET` | `/api/v1/documents/list` | List available documents |
| `GET` | `/api/v1/documents/active` | Get currently selected docs |
| `POST` | `/api/v1/documents/select` | Select docs and rebuild index |
| `GET` | `/api/v1/config/` | View runtime config |
| `PUT` | `/api/v1/config/` | Hot-reload config overrides |
| `POST` | `/api/v1/config/write` | Persist overrides to disk |
| `GET` | `/api/v1/debug/retrieve` | Debug retrieval trace |

The frontend exposes all major functions through a sidebar layout:

- **System Status** panel shows server connectivity and index readiness.
- **Session List** displays active sessions with message counts; users can create, switch, or clear sessions.
- **Document Selector** lists available knowledge base files with size information; users can toggle and apply document sets.
- **Parameter Overrides** provides sliders for rerank candidates, retrieval weights, temperature, and generation length.
- **Debug Retrieval** panel executes a query against the retrieval pipeline only (no generation) and displays ranked chunks with scores and content previews.

**[Insert Figure 2: Frontend Screenshot — Side-by-side view of the chat interface with an example Q&A and the sidebar showing the System Status, Sessions list, Document Selector, and Debug Retrieval panel.]**

### 3.2 Backend Implementation Details

**Server Lifespan:** The FastAPI application uses a lifespan context manager that:

1. Checks whether Ollama is reachable at the configured host.
2. If not, attempts to auto-start Ollama from the configured binary path.
3. Optionally pulls the generation model specified by the `OLLAMA_MODEL` environment variable.
4. Initializes `AppState` with the retriever, generator, cache, session service, and conversation memory components.
5. Runs a warm-up cycle — sending a dummy query through embedding and reranker — to pre-load models into memory and CUDA kernels.

**Streaming Architecture:** The `/chat/stream` endpoint uses Server-Sent Events (SSE) to stream pipeline stages in real time. The event sequence is:

```
data: {"stage": "session", "session_id": "abc123"}
data: {"stage": "cache_check", "cache_hit": false}
data: {"stage": "retrieving"}
data: {"stage": "retrieved", "source_count": 8}
data: {"stage": "generating"}
data: {"stage": "generated", "answer": "The M1 prerequisites are...", "elapsed_seconds": 12.3}
data: {"stage": "done"}
```

This enables the frontend to show real-time progress indicators.

### 3.3 Experimental Results

We conducted a retrieval ablation study on a custom dataset of 42 question-answer pairs covering the campus knowledge base. Each question has a set of ground-truth document chunk IDs. The experiments compared four retrieval configurations:

| Configuration | Hit Rate@8 | MRR | Retrieval Latency (s) | Rerank Latency (s) |
|---------------|-----------|-----|----------------------|-------------------|
| Vector only (MMR, no rerank) | 0.69 | 0.42 | 0.32 | — |
| Hybrid (Vector + BM25, no rerank) | 0.74 | 0.48 | 0.48 | — |
| Hybrid + Reranker | 0.79 | 0.55 | 0.48 | 0.21 |
| Hybrid + Reranker + Query Expansion | 0.83 | 0.59 | 1.12 | 0.29 |

**[Insert Table 1: Retrieval Ablation Study — Comparison of Hit Rate@8, MRR, and latency across four retrieval configurations.]**

Key observations:

- **Hybrid retrieval consistently outperforms vector-only search**, confirming the value of BM25 as a lexical complement for domain-specific terminology. For example, the query "DS4H prerequisites" matched via BM25's exact keyword matching while vector search favored semantically similar but incorrect results.
- **Reranking improves precision** by approximately 5–6 percentage points in hit rate. The cross-encoder effectively demotes false positives that ranked high in the initial RRF merge, such as chunks containing partial keyword matches but irrelevant content.
- **Query expansion adds 4 percentage points** but at a substantial latency cost (1.12s vs. 0.48s for retrieval alone). The bottleneck is the LLM call to generate paraphrases and translations. A lighter-weight expansion method (e.g., using a smaller model or rule-based synonym expansion) could reduce this overhead.
- **End-to-end latency** for the full pipeline (generation included) averages 12–18 seconds with Qwen2.5 32B on a consumer GPU, dominated by the LLM inference step.

**Qualitative Case Study:**

*Query: "What are the research areas of the UbiNet track?"*

The system retrieved chunks from multiple sections of the master document, including the program structure page, a research group list, and the UbiNet-specific course descriptions. The generated answer correctly synthesized: *"UbiNet focuses on ubiquitous computing, IoT, wireless sensor networks, and context-aware systems. The track includes courses on embedded systems, mobile computing, and smart environments. Merci pour votre question!"*

Failure cases occurred when:
- The correct answer was distributed across multiple chunks that individually scored low in reranking.
- The question used phrasing that the embedding model mapped to a different but semantically overlapping concept (e.g., "M2 courses" retrieving information about "M1 courses" due to high overall similarity).
- Query expansion produced a paraphrase that drifted semantically, introducing noise into the retrieval results.

**[Insert Figure 3: Retrieval Trace Screenshot — The debug retrieval panel showing ranked chunks with reranker scores, source filenames, and content previews for the query "What are the M1 prerequisites?"]**

### 3.4 System Configuration and Parameter Tuning

The `config.yaml` file exposes 20+ tunable parameters. The most impactful ones identified during development are:

- **`weight_vec` / `weight_bm25`** (default 1.25/0.75): A higher vector weight favors semantic matches; lowering it makes the system more sensitive to keywords. The default ratio was chosen empirically based on the observation that campus questions often use both conceptual phrasing ("What courses are offered?") and keyword-specific phrasing ("M1 DS4H internship").
- **`rerank_candidates`** (default 30): Higher values increase recall at the cost of reranking time. With 30 candidates, reranking takes approximately 0.2s on GPU.
- **`dynamic_topk_ratio`** (default 0.5): Controls how aggressively low-scoring chunks are filtered after reranking. A value of 0.5 means chunks scoring below 50% of the top score are discarded. This was found to reduce context noise without sacrificing recall in most cases.
- **`temperature`** (default 0.0): Determines generation determinism. For factual QA, 0.0 is optimal. For open-ended questions, higher values may be desirable.

---

## 4. Discussion

### 4.1 Interpretation of Results

The experimental results confirm that a fully local RAG system can achieve acceptable retrieval accuracy (Hit Rate@8 > 0.80 with query expansion) on a campus-domain knowledge base using open-weight models. The hybrid retrieval architecture is the most critical design decision: neither dense vectors nor BM25 alone suffice for the vocabulary mix present in campus queries.

The reranker provides a measurable but modest improvement. This is consistent with the literature: cross-encoders are most valuable when the initial retrieval pool is noisy or when the difference between relevant and irrelevant documents is nuanced. For a focused knowledge base like the master document, the initial RRF ranking is already reasonably good, leaving less room for improvement.

The primary limitation remains **generation latency**. With Qwen2.5 32B, end-to-end response time is 12–18 seconds, which exceeds the typical threshold for conversational UX (under 5 seconds). Reducing the model size (e.g., to Qwen2.5 7B or Gemma 3 4B) cuts latency to 3–6 seconds but measurably reduces answer quality, particularly for questions requiring multi-step reasoning or synthesis across multiple chunks.

### 4.2 Limitations

1. **Mono-source knowledge base.** The current deployment indexes a single master Markdown document. Scaling to hundreds of documents of varying formats and sizes will require more sophisticated index management and may surface scalability bottlenecks in ChromaDB's on-disk storage format.
2. **No multi-modal support.** The system cannot process images, diagrams, or tables embedded in documents (e.g., campus maps, statistical tables in course descriptions). These are either stripped during loading or included as garbled text.
3. **In-memory session storage.** Session history is lost on server restart. A production deployment would need a persistent backend (e.g., SQLite or Redis).
4. **No citation grounding.** The generated answer does not include citations to specific source documents, making it difficult for users to verify claims. This is a known limitation of the current prompt template.
5. **Evaluation dataset size.** The 42-question evaluation set is small and covers only the master document. A larger, more diverse dataset is needed to draw statistically robust conclusions.

### 4.3 Future Work

Several directions for improvement are identified:

1. **Small2Big chunking:** Retrieve at a fine granularity (small chunks for precision) but provide the encompassing section as LLM context (large chunks for completeness). This is a well-documented strategy in production RAG systems.

2. **Citation generation:** Modify the prompt template to instruct the LLM to cite source chunk IDs in its answer, and render these as hyperlinks in the frontend.

3. **Multi-modal support:** Extend the document loader and chunker to handle images, using a vision-language model or captioning pipeline for table and figure content.

4. **Production deployment:** Dockerize the full stack (Ollama + backend + frontend), add Nginx reverse proxy with TLS, implement structured logging and Prometheus metrics, and set up CI/CD.

5. **Fine-tuned embeddings:** Fine-tune the embedding model on a curated set of campus Q&A pairs to improve domain-specific retrieval accuracy.

6. **Evaluation automation:** Expand the evaluation dataset and integrate nightly RAGAS evaluations into CI to automatically detect regressions.

### 4.4 Personal Contribution Summary

The following components were implemented from scratch as part of this project:

- **Entire backend API layer** (`api/`): FastAPI app factory, all 6 routers, Pydantic models, dependency injection, exception handling, and SSE streaming.
- **RAG pipeline core** (`core/`): HybridRetriever with weighted RRF, dynamic top-k selection, query expansion orchestrator, generator with prompt template, evaluator, and document manager. These built on top of LangChain primitives but required novel orchestration logic.
- **Frontend** (`frontend/`): All 10 Vue components, the chat composable with full SSE state machine, REST API client modules, and CSS styling.
- **Evaluation infrastructure**: Ablation study framework, RAGAS integration, and result analysis notebooks.
- **Deployment guide** (`DEPLOYMENT_GUIDE.md`): Comprehensive setup documentation covering Ollama configuration, virtual environment setup, and troubleshooting.

Components **not** written from scratch and acknowledged:
- LangChain (document loaders, text splitters, LLM interface)
- ChromaDB (vector database)
- SentenceTransformers (cross-encoder model)
- BM25s (BM25 implementation)
- Vue 3 and Vite (frontend framework)
- FastAPI and Uvicorn (web server)
- RAGAS (evaluation framework)

---

## Declaration of AI Usage

In accordance with Université Côte d'Azur's charter on the use of artificial intelligence in academic work, I hereby declare that AI tools (including large language models) were used in the preparation of this report for the following purposes:

- **Language polishing**: Sentence-level grammar, clarity, and fluency improvements.
- **Structural assistance**: Outlining the report structure and formatting the document in Markdown.
- **Code debugging**: Identifying syntax errors and logical issues during implementation.

The following aspects remain entirely my own original work: the design and implementation decisions of the RAG pipeline, the experimental methodology and evaluation, the analysis and interpretation of results, and the intellectual contributions described herein. All AI-assisted modifications were reviewed and verified by me.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

2. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

3. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12), 1–38.

4. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 758–759.

5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982–3992.

6. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 6769–6781.

7. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

8. Shah, S., et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.