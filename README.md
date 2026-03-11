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
   pip install langchain langchain-community langchain-huggingface langchain-text-splitters langchain-ollama langchain-core chromadb
   ```
3. Ensure Ollama is installed and running with the `gemma3:1b` model:
   ```bash
   ollama pull gemma3:1b
   ```

## Usage

Place `master.md` in the same directory as `chat_box.py`.

Run the main script:
```bash
python chat_box.py
```
The chatbot will start an interactive loop. Type questions like "Quels sont les cours en semestre 1?" and press Enter. Type 'exit' to quit.

For command-line questions:
```bash
python chat_box.py -q "Your question here"
```

## Project Structure

- `chat_box.py`: Main script with data preparation, chatbot setup, and interactive loop.
- `master.md`: Markdown document containing information about the Master's program.

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

## Contributing

Feel free to submit issues or pull requests for improvements. Ensure changes align with the project's focus on RAG for educational Q&A.