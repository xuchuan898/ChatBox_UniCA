# Chatbox Project

Industrialized local RAG chatbot for Université Côte d'Azur CS Master Q&A.

## Key Upgrades

- Persistent indexing (`Chroma` + `BM25`) with hash validation and rebuild control.
- Modular pipeline (`document_loader`, `chunker`, `indexer`, `retriever`, `reranker`, `generator`).
- Query expansion module (`core/query_expander.py`) with pluggable rewriter:
  - `OllamaRewriter` (paraphrase + translation)
  - `PassthroughRewriter` (ablation baseline)
- Semantic cache (`core/semantic_cache.py`) with cosine threshold + TTL.
- Conversation memory (`core/conversation_memory.py`) for short/long-term context.
- Centralized configuration (`config.yaml`) with priority:
  - CLI > config.yaml > default values.

## Install

```bash
pip install langchain langchain-community langchain-huggingface langchain-text-splitters langchain-ollama langchain-core chromadb sentence-transformers bm25s mammoth pyyaml
```

Optional (PDF conversion):
```bash
pip install docling
```

Ensure Ollama model exists:
```bash
ollama pull gemma3:4b
```

## Build / Load Index

Offline build:
```bash
python build_index.py --doc-file ./docs/chroma/master.md
```

Runtime auto-logic in `chat_box.py`:
- meta exists + hash matches => load
- else => rebuild
- force via `--force-rebuild`

## Usage

Single question:
```bash
python chat_box.py --doc-file ./docs/chroma/master.md -q "What are M1 prerequisites?"
```

Batch questions:
```bash
python chat_box.py --doc-file ./docs/chroma/master.md --question-file ./questions/questions_batch_example.txt --answer-file ./docs/chroma/answers.txt
```

Enable query expansion controls:
```bash
python chat_box.py --enable-query-expansion true --expansion-model gemma3:4b --expansion-paraphrases 1 --expansion-add-translation true --expansion-source-lang auto
```

Enable cache controls:
```bash
python chat_box.py --enable-cache true --cache-threshold 0.92 --cache-ttl 86400
```

Enable memory controls:
```bash
python chat_box.py --enable-memory true --memory-rounds 5
```

## Project Structure

- `chat_box.py`: CLI orchestrator (memory -> cache -> expansion -> retrieve -> rerank -> generate).
- `build_index.py`: offline persistent index build.
- `core/config_loader.py`: config loading + CLI override merge.
- `core/document_loader.py`: MD/TXT/PDF/DOCX/URL loading.
- `core/chunker.py`: adaptive chunking.
- `core/indexer.py`: build/load/ensure persistent Chroma + BM25.
- `core/retriever.py`: hybrid retrieval + RRF + expansion aggregation.
- `core/reranker.py`: cross-encoder dual-score rerank.
- `core/generator.py`: constrained answer generation prompt.
- `core/query_expander.py`: query rewrite/translation + language detection.
- `core/semantic_cache.py`: semantic answer cache.
- `core/conversation_memory.py`: short/long-term memory.
- `core/evaluator.py`: context recall helper + RAGAS placeholder.
- `experiments/`: original experiment scripts (kept compatible).

## Notes on Compatibility

- Original `chat_box.py` exported APIs (`prepare_data`, `chatbox`) are preserved for `experiments/run_doc_matrix_experiment.py`.
- Existing CLI arguments remain valid; new features are opt-in or config-driven.
