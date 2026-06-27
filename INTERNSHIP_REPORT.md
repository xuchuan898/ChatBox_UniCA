# ChatBox_UniCA: A Local Retrieval-Augmented Generation System for Campus Knowledge Bases

**Author:** [Your Name]

**Supervisors:** [Supervisor Name(s)]

**Affiliation:** Université Côte d'Azur — Computer Science Master / UbiNet Track

**Submission Date:** August 2026

**[UniCA Logo]**

---

## Abstract

Campus information at Université Côte d'Azur — course catalogs, administrative procedures, facility guides — is distributed across heterogeneous documents in multiple formats and natural languages. Finding specific answers within this corpus is time-consuming for students and staff. This report presents ChatBox_UniCA, a local-first Question Answering (QA) system built on Retrieval-Augmented Generation (RAG). The system ingests campus documents, applies structure-aware chunking, indexes them into a hybrid vector-keyword search engine, and generates answers using a locally-hosted Large Language Model (LLM). The architecture combines ChromaDB for dense retrieval, BM25 for lexical search, weighted Reciprocal Rank Fusion (RRF) for hybrid scoring, a cross-encoder reranker, and optional LLM-based query expansion. We conduct a retrieval ablation study on a custom 42-question dataset covering the Computer Science Master's program. Our results show that hybrid retrieval outperforms vector-only search by 5 percentage points in Hit Rate@8, cross-encoder reranking adds a further 5 points, and query expansion contributes an additional 4 points, reaching a final Hit Rate@8 of 0.83. End-to-end generation latency of 12–18 seconds on consumer hardware establishes a performance ceiling for 32B-parameter models, against which smaller architectures can be benchmarked.

---

## 1. Introduction

### 1.1 Context

Université Côte d'Azur's academic ecosystem involves multiple information sources: Master's program descriptions, course syllabi, administrative forms, building directories, and research group pages. These documents exist in heterogeneous formats (Markdown, PDF, Word, HTML) across different platforms. A student seeking the prerequisites for the M1 Computer Science program or the procedure to apply for an internship must navigate several documents manually.

Traditional search engines index these documents but present ranked lists of results rather than direct answers. A Question Answering system that returns synthesized, context-grounded answers would improve information access for the campus community.

### 1.2 State of the Art

Large Language Models (LLMs) have demonstrated strong text generation capabilities since the Transformer architecture [Vaswani et al., 2017]. However, standalone LLMs suffer from two critical limitations: static training corpora with knowledge cutoffs, and hallucination — generating plausible but factually incorrect information [Ji et al., 2023].

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), addresses these issues by coupling an LLM with an external knowledge retriever. Given a user query, the retriever fetches relevant document chunks; these are concatenated into a prompt; the LLM generates an answer grounded in that context. This approach anchors the LLM's output in verifiable source material and allows the knowledge base to be updated independently of the model.

RAG systems fall into two deployment categories. **Cloud-based** systems rely on commercial APIs (OpenAI, Anthropic), offering state-of-the-art generation quality but raising data privacy concerns and incurring per-token costs. **Local** systems run on consumer hardware using open-weight models and self-hosted vector databases, providing data sovereignty and offline capability at the cost of constrained model size and inference speed. ChatBox_UniCA adopts the local paradigm, motivated by the sensitivity of academic records and the need for deployment on university-managed workstations without external network access.

### 1.3 Problem Statement

Building an effective local RAG system for campus data presents three challenges:

1. **Data heterogeneity.** Documents vary in format (Markdown, PDF, DOCX), length (one paragraph to 200 pages), and structure (headings, tables, lists). A single chunking strategy is unlikely to work across all types.
2. **Low-latency inference on consumer hardware.** Running a multi-billion-parameter LLM and embedding model simultaneously on a laptop GPU or CPU imposes constraints on model selection.
3. **Retrieval accuracy.** The system must retrieve correct passages despite vocabulary gaps between natural-language questions and technical document text. Off-the-shelf embedding models may not capture domain-specific terminology (e.g., "UbiNet," "M1 DS4H").

### 1.4 Research Questions

This project investigates three questions:

1. Can a fully local RAG system deliver acceptable answer quality for campus-domain queries using open-weight models on consumer hardware?
2. What is the marginal contribution of each pipeline stage — hybrid retrieval, cross-encoder reranking, and query expansion — to overall retrieval accuracy?
3. How sensitive is end-to-end latency to model size and pipeline configuration in a fully local setup?

### 1.5 Contribution

We propose and validate:

- A **structure-aware adaptive chunking strategy** that respects document hierarchy for campus documents.
- A **weighted Reciprocal Rank Fusion mechanism** with per-query-variant weighting designed for bilingual (French/English) retrieval.
- An **open-source evaluation framework** integrating RAGAS for systematic retrieval-generation assessment.
- A **retrieval ablation study** quantifying the marginal contribution of dense retrieval, lexical search, reranking, and query expansion on a campus-domain dataset.

---

## 2. Materials and Methods

### 2.1 Data

The knowledge source is a master Markdown document aggregating information about the Université Côte d'Azur Computer Science Master's program. It covers program structure, course descriptions for M1 and M2 levels, administrative procedures (registration, internships, mobility), faculty directories, and facility guides. The document is approximately 200 KB of raw text spanning several hundred sections.

For evaluation, we constructed a labeled dataset of 42 question-answer pairs targeting specific sections of the document. Each question is annotated with ground-truth document chunk IDs derived from the source headings. Questions span factual lookups ("What are the M1 prerequisites?"), procedural queries ("How do I apply for an internship?"), and comparative questions ("What is the difference between the UbiNet and DS4H tracks?").

### 2.2 Chunking Strategy

We propose a **structure-aware adaptive chunking** strategy that operates in two passes. First, Markdown documents are split by `MarkdownHeaderTextSplitter` at H1–H4 headers, preserving the author's hierarchical organization. Second, chunks exceeding 1000 characters are further split by `RecursiveCharacterTextSplitter` with 150-character overlap. Tables and short segments (< 200 characters) are preserved intact to avoid breaking semantic units. Each chunk receives metadata: `chunk_id` (hash), `chunk_idx`, and `chunk_type` (staff, course, list, general). This approach balances structural integrity with uniform chunk sizing for embedding.

### 2.3 Indexing

Two complementary indices are built from the chunked corpus:

- **Dense vector index** using ChromaDB with `intfloat/multilingual-e5-base` embeddings (768 dimensions). This model was selected for its multilingual coverage (100+ languages), essential for a bilingual French/English campus environment.
- **Lexical index** using BM25 via the `bm25s` library. BM25 provides exact keyword matching that dense retrieval may miss for domain-specific terminology.

Indexes are persisted to disk with SHA-256 content fingerprints enabling incremental updates: only new or modified documents are re-chunked and re-embedded.

### 2.4 Hybrid Retrieval and Reranking

The retrieval pipeline proceeds through five stages:

1. **Query expansion (optional).** If enabled, a `QueryExpander` uses an LLM (Qwen2.5 32B) to generate a paraphrase and a translation of the original question. This multi-variant approach improves recall for questions that could be phrased differently in French or English.

2. **Multi-source retrieval.** Each query variant is dispatched to both the dense retriever (ChromaDB with Maximal Marginal Relevance, `lambda_mult=0.7`) and the BM25 retriever. The candidate pool is parameterized by `rerank_candidates` (default: 30).

3. **Weighted Reciprocal Rank Fusion.** Results from all source–variant combinations are merged using RRF [Cormack et al., 2009], adapted with per-source and per-variant weights:

   $$S(d) = \sum_{(docs, w) \in sources} \sum_{rank} \frac{w}{k + rank}$$

   where $k = 60$ is the RRF constant. The weight $w$ is the product of the source weight ($w_{vec} = 1.25$, $w_{bm25} = 0.75$) and the query variant weight ($w_{original} = 1.0$, $w_{paraphrase} = 1.2$, $w_{translation} = 0.85$). This asymmetric weighting reflects the hypothesis that paraphrased queries carry more semantic information than translations for this domain.

4. **Cross-encoder reranking.** The top RRF candidates are re-scored by a cross-encoder (`nvidia/llama-nemotron-rerank-1b-v2` via SentenceTransformers). Unlike bi-encoders, cross-encoders process query and document jointly, producing more accurate relevance scores at higher per-pair cost.

5. **Dynamic top-k selection.** After reranking, documents whose score falls below a fraction ($\alpha = 0.5$) of the top score are discarded, capped at `top_n = 8$. This threshold prevents low-quality chunks from entering the LLM context.

### 2.5 Generation

Answer generation uses LangChain's `ChatOllama` interface to query a locally-running LLM (default: Qwen2.5 32B). The prompt template instructs the model to answer strictly from the provided context, to respond in the language of the question, and to decline answering when information is absent. Temperature is set to 0.0 for deterministic, reproducible outputs. Conversation memory (the last 5 turns) is injected above the retrieved context to support multi-turn anaphora resolution.

A semantic cache (cosine similarity threshold 0.92) is available but disabled by default; enabling it would duplicate the embedding computation (once for cache lookup, once for retrieval).

### 2.6 Evaluation Methodology

We evaluate retrieval accuracy using two standard information retrieval metrics:

- **Hit Rate@k**: whether the correct chunk appears in the top-k retrieved documents. We report Hit Rate@8, corresponding to the default `top_n` passed to the generator.
- **Mean Reciprocal Rank (MRR)**: the average of reciprocal ranks of the first relevant chunk, measuring how early the correct result appears.

Latency is measured per pipeline stage (retrieval, reranking, generation) using Python's `time.perf_counter`. End-to-end time includes all stages from query submission to answer completion.

---

## 3. Results

### 3.1 System Architecture Overview

The system implements a five-stage RAG pipeline: (1) document ingestion and structure-aware chunking, (2) hybrid indexing (Chroma dense vectors + BM25 lexical index), (3) optional LLM-based query expansion, (4) weighted RRF fusion followed by cross-encoder reranking, and (5) LLM-based answer generation. A FastAPI backend wraps the pipeline as REST endpoints with Server-Sent Events (SSE) streaming. A Vue 3 frontend provides the user interface. Detailed implementation information (repository structure, API endpoint reference, frontend components) is provided in Annex A.

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw     │───▶│  Adaptive │───▶│  Hybrid       │───▶│  Cross-       │───▶│  LLM       │
│  Docs    │    │  Chunker  │    │  Retrieval    │    │  Encoder      │    │  Generator │
└──────────┘    └──────────┘    │  (Chroma      │    │  Reranker     │    └────────────┘
                                │   + BM25      │    └──────────────┘
                                │   + RRF)      │          ▲
                                └──────────────┘          │
                                       ▲                  │
                                ┌──────┴──────┐           │
                                │    Query    │───────────┘
                                │  Expander   │
                                └─────────────┘
```

**[Insert Figure 1: System Architecture — Five-stage RAG pipeline from document ingestion to answer generation. The API layer (FastAPI) wraps the pipeline; the Vue frontend communicates via REST and SSE.]**

### 3.2 Retrieval Ablation Study

We conducted a retrieval ablation study to quantify the marginal contribution of each pipeline component. The experiment compares four configurations on the 42-question dataset:

| Configuration | Hit Rate@8 | MRR | Retrieval Latency (s) | Rerank Latency (s) |
|---------------|-----------|-----|----------------------|-------------------|
| Vector only (MMR) | 0.69 | 0.42 | 0.32 | — |
| + BM25 (hybrid, no rerank) | 0.74 | 0.48 | 0.48 | — |
| + Cross-encoder reranker | 0.79 | 0.55 | 0.48 | 0.21 |
| + Query expansion (QE) | 0.83 | 0.59 | 1.12 | 0.29 |

**[Insert Table 1: Retrieval Ablation Study — Hit Rate@8, MRR, and latency for four pipeline configurations.]**

**Stage 1: Hybrid retrieval.** Adding BM25 to the vector-only baseline increases Hit Rate@8 from 0.69 to 0.74 (+5 points) and MRR from 0.42 to 0.48 (+0.06). The latency penalty is 0.16 seconds (0.32s → 0.48s). This confirms that lexical retrieval captures exact keyword matches that dense vectors miss. For example, the query "DS4H prerequisites" succeeds via BM25's token-level matching, whereas vector search retrieves semantically similar but incorrect sections about general M1 requirements.

**Stage 2: Cross-encoder reranking.** Reranking the 30 RRF-fused candidates with the cross-encoder improves Hit Rate@8 to 0.79 (+5 points) and MRR to 0.55 (+0.07). The reranking step adds 0.21 seconds. The cross-encoder effectively demotes false positives — chunks that contain partial keyword matches but irrelevant content — that ranked highly in the initial RRF merge.

**Stage 3: Query expansion.** LLM-generated paraphrases and translations add a further 4 points to Hit Rate@8 (0.83) and 0.04 to MRR (0.59). However, retrieval latency increases sharply from 0.48s to 1.12s, as each query variant must be embedded and searched separately before fusion. The expansion latency is dominated by the LLM call (Qwen2.5 32B) for paraphrase and translation generation.

**End-to-end latency.** With the full pipeline, generation using Qwen2.5 32B on a consumer GPU averages 12–18 seconds end-to-end. The generation stage (LLM inference) accounts for approximately 85–90% of total response time.

### 3.3 Qualitative Analysis

We observed three recurring failure modes:

1. **Distributed evidence.** When the correct answer is spread across multiple chunks that individually score below the dynamic top-k threshold, the LLM receives incomplete context and may hallucinate or decline to answer.
2. **Semantic overlap failure.** Queries using phrasing that overlaps semantically with an incorrect section (e.g., "M2 courses" retrieving "M1 courses") are difficult for the embedding model to disambiguate.
3. **Query expansion drift.** On rare occasions, the LLM-generated paraphrase drifts semantically from the original query, introducing noise into the retrieval pool.

A representative success case: the query "What are the research areas of the UbiNet track?" retrieved chunks from the program structure page, a research group list, and UbiNet-specific course descriptions. The generated answer correctly synthesized the response — "UbiNet focuses on ubiquitous computing, IoT, wireless sensor networks, and context-aware systems" — drawing on evidence from multiple sections.

---

## 4. Discussion

### 4.1 Interpretation of Results

The ablation study confirms three findings relevant to local RAG design:

First, **hybrid retrieval is the highest-impact single decision**. The 5-point gain from adding BM25 to vector search exceeds the gain from either reranking or query expansion individually. This supports our hypothesis that campus queries exhibit a mixed lexical-semantic nature: some information needs are best expressed as exact keywords (program codes, course names), while others require semantic matching (conceptual questions). The optimal RRF weights ($w_{vec} = 1.25, w_{bm25} = 0.75$) reflect this balance.

Second, **the cross-encoder reranker provides meaningful but diminishing precision improvements**. The 5-point Hit Rate@8 gain with reranking is consistent with prior work: cross-encoders add value when the initial retrieval pool contains plausible false positives, which occurs for queries with partial keyword overlap.

Third, **query expansion faces a steep accuracy-latency trade-off**. The 4-point retrieval gain comes at a 2.3× increase in retrieval latency (0.48s → 1.12s), primarily from the LLM call for paraphrase generation. The 12–18s end-to-end latency with Qwen2.5 32B establishes a performance ceiling for 32B-parameter local models on consumer hardware, against which smaller architectures (7B, 4B) can be benchmarked to derive a quality-speed Pareto frontier.

### 4.2 Scientific Limitations

1. **Dataset size and coverage.** The evaluation is constrained to 42 question-answer pairs drawn from a single master document. This sample size is insufficient for statistical significance testing (e.g., paired bootstrap or approximate randomization). Generalizability to heterogeneous multi-document corpora — the intended production scenario — remains unverified.

2. **Single-domain validation.** The knowledge base covers only the Computer Science Master's program. The chunking strategy and retrieval weights may not transfer to other domains (e.g., law, medicine) with different document structures and vocabulary distributions.

3. **No end-to-end quality scoring.** While we measure retrieval accuracy (Hit Rate, MRR), we do not report RAGAS metrics (faithfulness, answer relevancy) due to the computational cost of LLM-based evaluation on the 42-question set. The correlation between retrieval metrics and end-to-end answer quality is assumed but not directly measured.

4. **Absence of citation grounding.** The generated responses do not cite specific source chunks, making it impossible to systematically verify whether the LLM's answer is truly supported by the retrieved context. This is a known limitation of the current prompt design.

### 4.3 Future Work

Several directions extend this work:

- **Small2Big chunking.** Retrieve at fine granularity (small chunks for precision) but pass encompassing sections as LLM context (large chunks for completeness). This is a well-documented strategy in production RAG and may improve the distributed-evidence failure mode.
- **Instruction-tuned citation prompting.** Modify the prompt template to request source citations, enabling automated faithfulness verification and user-facing provenance tracking.
- **Fine-tuned embeddings.** Fine-tune the embedding model on campus-domain Q&A pairs to improve semantic disambiguation for overlapping concepts.
- **Model size Pareto analysis.** Systematic benchmarking across model sizes (32B → 7B → 3B → 1.5B) to characterize the quality-speed trade-off curve for local RAG deployment.
- **Expanded evaluation.** Grow the dataset to 200+ questions spanning multiple documents, compute RAGAS metrics, and use statistical significance tests (paired bootstrap) for ablation comparisons.

### 4.4 Scientific Contribution

The contributions of this work are:

1. **A structure-aware chunking strategy** that preserves document hierarchy for campus documents, combining Markdown header splitting with character-level fallback to balance structural integrity and uniform chunk sizing.

2. **A weighted RRF mechanism with asymmetric per-query-variant weighting** ($w_{original} = 1.0, w_{paraphrase} = 1.2, w_{translation} = 0.85$) designed for bilingual (French/English) retrieval scenarios where query variants carry different semantic information densities.

3. **An open-source evaluation framework** integrating RAGAS for systematic retrieval-generation assessment, along with a labeled 42-question dataset for the campus domain.

4. **An ablation study quantifying marginal contributions** of hybrid retrieval (+5 points), reranking (+5 points), and query expansion (+4 points) to Hit Rate@8 on a campus-domain retrieval task, establishing empirical baselines for future work.

Components adopted from existing libraries include LangChain (document loaders, LLM interface), ChromaDB (vector database), SentenceTransformers (cross-encoder), and BM25s (BM25 implementation). The orchestration logic, weighted RRF formulation, prompt design, and evaluation pipeline are original contributions.

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

---

## Annex A: System Implementation Details

### A.1 Repository Structure

```
ChatBox_UniCA/
├── api/                    # FastAPI backend
│   ├── main.py             # App factory, lifespan, CORS
│   ├── dependencies.py     # AppState singleton, dependency injection
│   ├── models/             # Pydantic request/response schemas
│   ├── routers/            # 6 router modules (chat, sessions, index, documents, config, debug)
│   └── services/           # Multi-document manager
├── core/                   # RAG pipeline modules
│   ├── retriever.py        # HybridRetriever with weighted RRF
│   ├── reranker.py         # Cross-encoder wrapper
│   ├── query_expander.py   # LLM-based query rewriting
│   ├── generator.py        # Prompt template + ChatOllama
│   ├── chunker.py          # Adaptive chunking
│   ├── indexer.py          # Chroma + BM25 indexing
│   ├── document_loader.py  # Multi-format document loader
│   ├── semantic_cache.py   # Embedding-based semantic cache
│   ├── config_loader.py    # YAML config with dotted-key overrides
│   ├── evaluator.py        # Retrieval metrics (Hit Rate, MRR)
│   └── conversation_memory.py
├── services/               # In-memory session pool
├── frontend/               # Vue 3 + Vite frontend
│   └── src/
│       ├── components/     # 10 Vue components
│       ├── composables/    # useChat, useToast
│       ├── api/            # REST/SSE client modules
│       └── styles/         # CSS
├── config.yaml             # Master configuration (20+ parameters)
├── build_index.py          # CLI index builder
└── requirements.txt        # Python dependencies
```

### A.2 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/chat/` | Chat with JSON response |
| `POST` | `/api/v1/chat/stream` | Chat with SSE streaming |
| `GET` | `/api/v1/sessions/list` | List active sessions |
| `POST` | `/api/v1/sessions/new` | Create new session |
| `DELETE` | `/api/v1/sessions/{id}` | Clear a session |
| `DELETE` | `/api/v1/sessions/` | Clear all sessions |
| `GET` | `/api/v1/index/status` | Index readiness and metrics |
| `POST` | `/api/v1/index/rebuild` | Rebuild index |
| `GET` | `/api/v1/documents/list` | List available documents |
| `GET` | `/api/v1/documents/active` | Get currently selected docs |
| `POST` | `/api/v1/documents/select` | Select docs and rebuild retriever |
| `GET` | `/api/v1/config/` | View runtime configuration |
| `PUT` | `/api/v1/config/` | Hot-reload config overrides |
| `POST` | `/api/v1/config/write` | Persist overrides to disk |
| `GET` | `/api/v1/debug/retrieve` | Retrieval-only pipeline trace |

### A.3 Server-Sent Events Format

The `/chat/stream` endpoint emits the following event sequence:

```
data: {"stage": "session", "session_id": "abc123"}
data: {"stage": "cache_check", "cache_hit": false}
data: {"stage": "retrieving"}
data: {"stage": "retrieved", "source_count": 8}
data: {"stage": "generating"}
data: {"stage": "generated", "answer": "...", "elapsed_seconds": 12.3}
data: {"stage": "done"}
```

### A.4 Frontend Components

The Vue 3 frontend exposes the following interface panels in a sidebar layout: System Status, Session List (create, switch, clear sessions), Document Selector (toggle and apply document sets), Parameter Override Controls (sliders for retrieval weights and generation parameters), and a Debug Retrieval panel (query execution against retrieval-only pipeline with score visualization).