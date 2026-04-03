# Chatbox Project

This project implements a simple chatbot for answering questions about the Master's program in Computer Science at Université Côte d'Azur, using retrieval-augmented generation (RAG) with LangChain, Chroma vector database, HuggingFace embeddings, and an Ollama-hosted LLM.

## Features

- Loads and processes the document from `master.md`.
- Splits text using Markdown headers and token-based chunking.
- Creates an in-memory Chroma vector database for efficient retrieval.
- Uses Maximal Marginal Relevance (MMR) for diverse document retrieval.
- Interactive chat interface in the terminal.
- Answers in the language of the query, concisely, and ends with "Merci pour votre question!"

## Installation

1. Clone or download the project.
2. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-huggingface langchain-text-splitters langchain-ollama langchain-core chromadb sentence-transformers bm25s pymupdf4llm mammoth
   ```
3. Ensure Ollama is installed and running with the `gemma3:1b` model:
   ```bash
   ollama pull gemma3:1b
   ```

## Usage

Run with a single local document (fastest):
```bash
python chat_box.py --doc-file ./docs/chroma/master.txt
```

Optional URL sources (one URL per line in `./docs/chroma/source_urls.txt`):
```bash
python chat_box.py --doc-file ./docs/chroma/master.txt --url-file ./docs/chroma/source_urls.txt
```

Single question mode:
```bash
python chat_box.py --doc-file ./docs/chroma/master.txt -q "Your question here"
```

Batch question mode (`./questions/*.txt`):
```bash
python chat_box.py --doc-file ./docs/chroma/master.txt --question-file ./questions/questions_batch_example.txt --answer-file ./docs/chroma/answers.txt
```

Run full question x document matrix experiment (exports prompts + answers for later AI analysis):
```bash
python experiments/run_doc_matrix_experiment.py --question-catalog ./questions/generated_questions_docs_chroma.json
```

Debug mode (timing, chunks, retrieval details):
```bash
python chat_box.py --doc-file ./docs/chroma/master.txt --debug
```

## Project Structure

- `chat_box.py`: Main script with data preparation, chatbot setup, and interactive loop.
- `docs/chroma/master.md`: Original markdown knowledge document.
- `docs/chroma/master.txt`: Plain-text version used for format comparison experiments.
- `docs/chroma/source_urls.txt`: Optional URL list for web/calendar ingestion.
- `questions/questions_batch_example.txt`: Example batch question file.
- `questions/generated_questions_docs_chroma.json`: 5 curated questions per document in `docs/chroma` (excluding `source_urls.txt`).
- `experiments/run_doc_matrix_experiment.py`: Runs all questions against all documents and exports prompts/retrieval/answers.

## How It Works

1. **Data Preparation**: Loads `master.md`, splits by Markdown headers, then into chunks of 400 tokens with 120 overlap. Enhances M1-related chunks for better retrieval.
2. **Embeddings**: Uses `intfloat/multilingual-e5-base` for vectorization.
3. **Retrieval**: MMR retriever fetches 10 relevant docs from 20 candidates, with optional metadata filtering for M1 queries.
4. **Generation**: Ollama's Gemma3:1b generates answers based on context and prompt.
5. **Chat Loop**: Processes user input, retrieves context, generates response.

## Dependencies

- Python 3.x
- LangChain ecosystem
- ChromaDB
- HuggingFace Transformers
- Ollama

## Matrix Experiment Outputs

`experiments/run_doc_matrix_experiment.py` creates a timestamped folder in `experiments/results/` with:

- `responses.jsonl`: one row per (question, target doc), including rendered prompt, retrieval trace, answer, and timing.
- `prompts.jsonl`: prompt-focused export for downstream AI analysis.
- `comparison.csv`: side-by-side answers per question across all target docs.
- `summary.json` and `summary.md`: aggregate timing and answer-shape metrics.

## Contributing

Feel free to submit issues or pull requests for improvements. Ensure changes align with the project's focus on RAG for educational Q&A.
