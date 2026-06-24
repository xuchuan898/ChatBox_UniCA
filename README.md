
# Chatbox Project

Industrialized local RAG chatbot for Université Côte d'Azur CS Master Q&A.

## Key Upgrades

- Persistent indexing (Chroma + BM25) with hash validation and rebuild control.
- Modular pipeline (document_loader, chunker, indexer, retriever, reranker, generator).
- Query expansion module (`core/query_expander.py`) with pluggable rewriter:
  - `OllamaRewriter` (paraphrase + translation)
  - `PassthroughRewriter` (ablation baseline)
- Semantic cache (`core/semantic_cache.py`) with cosine threshold + TTL.
- Conversation memory (`core/conversation_memory.py`) for short/long-term context.
- Centralized configuration (`config.yaml`) with priority:
  - CLI > config.yaml > default values.

---

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

---

## Build / Load Index

Offline build:

```bash
python build_index.py --doc-file ./docs/chroma/master.md
```

Runtime auto-logic in `chat_box.py`:

- meta exists + hash matches => load
- else => rebuild
- force via `--force-rebuild`

---

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

---

## Project Structure

```
chat_box.py          # CLI orchestrator (memory -> cache -> expansion -> retrieve -> rerank -> generate)
build_index.py       # offline persistent index build
core/
├── config_loader.py         # config loading + CLI override merge
├── document_loader.py       # MD/TXT/PDF/DOCX/URL loading
├── chunker.py               # adaptive chunking
├── indexer.py               # build/load/ensure persistent Chroma + BM25
├── retriever.py             # hybrid retrieval + RRF + expansion aggregation
├── reranker.py              # cross-encoder dual-score rerank
├── generator.py             # constrained answer generation prompt
├── query_expander.py        # query rewrite/translation + language detection
├── semantic_cache.py        # semantic answer cache
├── conversation_memory.py   # short/long-term memory
├── evaluator.py             # context recall helper + RAGAS placeholder
experiments/                 # original experiment scripts (kept compatible)
```

---

## Notes on Compatibility

- Original `chat_box.py` exported APIs (`prepare_data`, `chatbox`) are preserved for `experiments/run_doc_matrix_experiment.py`.
- Existing CLI arguments remain valid; new features are opt-in or config-driven.

---

# ChatBox_UniCA API Server

FastAPI-based RAG backend service providing RESTful and SSE streaming endpoints.

## Prerequisites

- Python 3.10+
- Conda environment (recommended) or system Python:

```bash
module load conda                    # HPC cluster only
conda activate /path/to/chatbox_env  # or: conda activate chatbox
```

- Ollama binary in PATH or at `~/ollama/bin/ollama`:

```bash
export PATH=$PATH:~/ollama/bin
ollama pull qwen2.5:32b
```

- Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Index Building

Build the index before the first run (skipped if metadata already exists under `./index_store`):

```bash
python build_index.py --doc-file ./docs/chroma/master.md --persist-dir ./index_store
```

---

## Starting the API Server

The server will auto-start Ollama if it is not already running (looks for the binary in PATH and `~/ollama/bin/`), then ensure the configured model exists, and finally load indexes and models.

> **Note**: `uvicorn --factory` may cause Segmentation fault due to CUDA/torch fork conflicts. Use the Python‑script method below to avoid this issue.

```bash
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

Or set `USER_AGENT` to suppress the warning:

```bash
USER_AGENT=MyBot/1.0 python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

First startup loads models and indexes, which may take 10–30 seconds. Ollama auto-start waits up to 30s.

---

## Environment Variables (all optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_BIN_DIR` | `~/ollama/bin` | Directory to find ollama binary |
| `OLLAMA_LOG_FILE` | `~/ollama/ollama.log` | Path to Ollama server log |
| `OLLAMA_MODEL` | (not set) | Model to pull on startup (e.g. `qwen2.5:32b`) |
| `OLLAMA_STARTUP_WAIT` | `30` | Seconds to wait for Ollama to become ready |

If you prefer to manage Ollama yourself, start it before the API server and the auto-start logic will skip.

---

## API Endpoints

- Swagger Docs: `http://<server-ip>:8000/docs`
- Health Check: `GET /health`

---

## Testing with Apifox

1. Open Apifox and create a new project.
2. Go to **Import** → **Import Swagger/OpenAPI Document**.
3. Enter URL: `http://<server-ip>:8000/openapi.json` — Apifox will parse all endpoints automatically.
4. Set an environment variable `base_url` → `http://<server-ip>:8000` (or just use direct URLs).
5. Now you can debug every endpoint from the Apifox interface.

---

## Chat (JSON)

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are M1 prerequisites?", "session_id": "test-001"}'
```

---

## Chat (SSE Stream)

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the internship process"}'
```

---

## Index Status

```bash
curl http://localhost:8000/api/v1/index/status
```

---

## Dynamic Parameter Overrides

Pass `overrides` in the request body to override config parameters at runtime:

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "session_id": "test-001",
    "overrides": {
      "top_k": 10,
      "retrieval.weight_vec": 1.5,
      "retrieval.weight_bm25": 0.5,
      "retrieval.rerank_alpha": 0.3,
      "retrieval.rerank_candidates": 20
    }
  }'
```

---

## Available Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check |
| `GET` | `/` | API info |
| `POST` | `/api/v1/chat/` | Chat (JSON response) |
| `POST` | `/api/v1/chat/stream` | Chat (SSE stream) |
| `POST` | `/api/v1/sessions/new` | Create a new session |
| `DELETE` | `/api/v1/sessions/{session_id}` | Clear session memory |
| `GET` | `/api/v1/index/status` | Index status |
| `POST` | `/api/v1/index/rebuild` | Rebuild index (async) |
| `POST` | `/api/v1/index/update` | Incremental index update |
| `GET` | `/api/v1/config/` | View config (sanitized) |
| `PUT` | `/api/v1/config/` | Hot-reload config |

---

## Project Structure (added files)

```
api/
├── models/
│   ├── request.py       # Pydantic request models
│   └── response.py      # Pydantic response models
├── routers/
│   ├── chat.py          # Chat endpoints (JSON + SSE)
│   ├── sessions.py      # Session management
│   ├── index.py         # Index status / rebuild / update
│   └── config.py        # Config query and hot-reload
├── dependencies.py      # Global singleton dependency injection
├── exceptions.py        # Custom exceptions
└── main.py              # FastAPI app factory

services/
├── session_service.py   # In-memory session pool
└── rag_service.py       # Full RAG pipeline orchestration
```

---

## Notes

- **Port**: Ensure port 8000 is open in your firewall / security group.
- **Single worker**: Use `--workers 1` to avoid duplicate model loading across processes.
- **Cache**: Semantic cache is disabled by default. Enable it in `config.yaml` (`cache.enabled: true`).
- **Immutable core**: Do not modify any code under `core/` or `experiments/`.

---

## Remote Debugging & Common Issues (Grid'5000 Environment)

### 1. Network Topology and Tunnel Setup

**Login path**: `local machine → ${JUMP_HOST} → ${INTERMEDIATE_HOST} → ${TARGET_HOST} (compute node)`

**Variable definitions**:

| Variable | Meaning | Example Value | How to Obtain |
| :--- | :--- | :--- | :--- |
| `${USERNAME}` | Grid'5000 username | `bma` | Your login account |
| `${JUMP_HOST}` | Jump host address | `access.grid5000.fr` | Fixed value |
| `${INTERMEDIATE_HOST}` | Intermediate node (short name) | `sophia` | After logging into the jump host, run `hostname` or check the prompt |
| `${TARGET_HOST}` | Compute node (short name) | `esterel34-1` | After logging into the intermediate node, run `hostname` |
| `${LOCAL_PORT}` | Local mapped port | `8000` | Any available port; should match the remote port |
| `${REMOTE_PORT}` | Remote service port | `8000` | The port used when starting FastAPI |

**Generic tunnel command** (replace variables):

```bash
ssh -L ${LOCAL_PORT}:${TARGET_HOST}:${REMOTE_PORT} \
    -o ProxyCommand="ssh -W %h:%p ${USERNAME}@${JUMP_HOST}" \
    ${USERNAME}@${INTERMEDIATE_HOST}
```

**Example (with a real setup)**:

Assume your username is `bma`, jump host is `access.grid5000.fr`, intermediate node is `sophia`, and compute node is `esterel34-1`:

```bash
ssh -L 8000:esterel34-1:8000 \
    -o ProxyCommand="ssh -W %h:%p bma@access.grid5000.fr" \
    bma@sophia
```

> **Replace `esterel34-1` and `sophia` with your actual node names.**

- You will be asked for **two passwords** (jump host password + intermediate node password).
- The terminal will hang — this is **normal**; it means the tunnel is active.
- **Important**: Do not close this terminal window; it will break the tunnel.

**Verify the tunnel** (in another local terminal):

```bash
curl http://127.0.0.1:${LOCAL_PORT}/health
```

Expected output:

```json
{"status":"ok","service":"ChatBox_UniCA"}
```

**How to find your node names**:

```bash
# 1. Log in to the jump host
ssh ${USERNAME}@${JUMP_HOST}

# 2. Jump to the intermediate node (e.g., sophia)
ssh ${INTERMEDIATE_HOST}

# 3. Check the current compute node name
hostname
# Example output: esterel34-1 (this is your ${TARGET_HOST})
```

---

### 2. Starting the Service (on the compute node ${TARGET_HOST})

**Generic command**:

```bash
cd ~/ChatBox_UniCA
conda activate chatbox
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=${REMOTE_PORT})"
```

**Example** (using port 8000):

```bash
cd ~/ChatBox_UniCA
conda activate chatbox
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

**Success indicator**:

```
INFO:     Uvicorn running on http://0.0.0.0:${REMOTE_PORT} (Press CTRL+C to quit)
INFO:     Application startup complete.
```

**Important notes**:

- You must bind to `host='0.0.0.0'`, otherwise the tunnel cannot forward traffic.
- If you get a `Segmentation fault`, use the `python -c` method instead of `uvicorn --factory`.

---

### 3. Apifox Configuration

**Base URL (environment variable)**:

```
http://127.0.0.1:${LOCAL_PORT}
```

**Example** (port 8000):

```
http://127.0.0.1:8000
```

**Do NOT use**:

- ❌ Public IP (firewall blocks it)
- ❌ Jump host address (no service there)
- ❌ SSH format like `bma@...`

**Import API documentation**:

1. Click **Import** → **OpenAPI/Swagger**.
2. Enter URL: `http://127.0.0.1:${LOCAL_PORT}/openapi.json` (example: `http://127.0.0.1:8000/openapi.json`).
3. All endpoints will be generated automatically.

**Timeout setting**:

- Apifox settings → timeout → set to at least `120000ms` (120 seconds).
- Ollama inference can be slow (especially with large models).

---

### 4. SSE Streaming Endpoint Notes

- **Endpoint**: `POST /api/v1/chat/stream`
- **View in Apifox**: After sending the request, look for the **"EventStream"** or **"SSE Preview"** tab in the **bottom-right** corner; do not look at the regular Response Body.
- **Alternative**: Use `curl -N` in a local terminal:

**Generic command**:

```bash
curl -N -X POST http://127.0.0.1:${LOCAL_PORT}/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the M1 director?", "session_id": "test"}'
```

**Example** (port 8000):

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the M1 director?", "session_id": "test"}'
```

---

### 5. Troubleshooting Checklist

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `curl http://127.0.0.1:${LOCAL_PORT}/health` times out | Tunnel not established or service not running | Check that the tunnel terminal is still active; check the service terminal for `Uvicorn running` log |
| Apifox shows `ETIMEDOUT` | Apifox is using a public IP or wrong address | Change to `http://127.0.0.1:${LOCAL_PORT}` (example: `http://127.0.0.1:8000`) |
| Request hangs > 2 minutes | Ollama inference is slow (e.g., 32B model) | Add `"enable_query_expansion": false` in `overrides` to skip rewriting; or switch to a smaller model in `config.yaml` (e.g., `gemma3:4b`) |
| No `EventStream` tab in Apifox | Apifox version does not support SSE | Use the browser Swagger UI at `http://127.0.0.1:${LOCAL_PORT}/docs` (example: `http://127.0.0.1:8000/docs`) |
| `channel 3: open failed: connect failed: Connection refused` | Tunnel connected to wrong node (e.g., jump host instead of compute node) | Re‑run the tunnel command ensuring the target is `${TARGET_HOST}` (compute node) |
| `Unrecognized keys in 'rope_scaling'` at startup | Model config compatibility warning | Ignore; it does not affect functionality |

---

### 6. Quick Smoke Test for All Endpoints

Run these commands in a local terminal (keep the tunnel running):

**Generic commands (with variables)**:

```bash
# 1. Health check
curl http://127.0.0.1:${LOCAL_PORT}/health

# 2. Index status
curl http://127.0.0.1:${LOCAL_PORT}/api/v1/index/status

# 3. Chat (JSON)
curl -X POST http://127.0.0.1:${LOCAL_PORT}/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the M1 director?", "session_id": "test"}'

# 4. Streaming chat (SSE)
curl -N -X POST http://127.0.0.1:${LOCAL_PORT}/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What courses in S1?", "session_id": "test"}'

# 5. Create a new session
curl -X POST http://127.0.0.1:${LOCAL_PORT}/api/v1/sessions/new
```

**Example (port 8000)**:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Index status
curl http://127.0.0.1:8000/api/v1/index/status

# Chat
curl -X POST http://127.0.0.1:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the M1 director?", "session_id": "test"}'

# Streaming
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What courses in S1?", "session_id": "test"}'

# New session
curl -X POST http://127.0.0.1:8000/api/v1/sessions/new
```

---

### 7. Common Maintenance Commands (run on the server)

```bash
# Check service process
ps aux | grep uvicorn | grep -v grep

# Check if ${REMOTE_PORT} is listening
netstat -tlnp | grep ${REMOTE_PORT}

# Kill the service process (restart)
kill -9 <PID>

# Check Ollama status
curl http://127.0.0.1:11434/api/tags

# Test API locally (no tunnel needed)
curl http://127.0.0.1:${REMOTE_PORT}/health
```

**Example** (port 8000):

```bash
netstat -tlnp | grep 8000
curl http://127.0.0.1:8000/health
```

---

### 8. Placeholder Variable Reference Table

| Variable | Meaning | Example Value (in this guide) | How to Obtain |
| :--- | :--- | :--- | :--- |
| `${USERNAME}` | Grid'5000 username | `bma` | Your login account |
| `${JUMP_HOST}` | Jump host address | `access.grid5000.fr` | Fixed value |
| `${INTERMEDIATE_HOST}` | Intermediate node (short name) | `sophia` | After logging into the jump host, run `hostname` or check the prompt |
| `${TARGET_HOST}` | Compute node (short name) | `esterel34-1` | After logging into the intermediate node, run `hostname` |
| `${LOCAL_PORT}` | Local mapped port | `8000` | Any available port; should match the remote port |
| `${REMOTE_PORT}` | Remote service port | `8000` | The port used when starting FastAPI |
```