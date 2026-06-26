# ChatBox_UniCA — Local Deployment Guide

Deploy and run the ChatBox_UniCA RAG chatbot (backend + frontend) on your local laptop.

---

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Clone the Repository](#2-clone-the-repository)
- [3. Setup Ollama (LLM Backend)](#3-setup-ollama-llm-backend)
- [4. Setup Python Backend](#4-setup-python-backend)
- [5. Setup Vue Frontend](#5-setup-vue-frontend)
- [6. Run the Backend Server](#6-run-the-backend-server)
- [7. Run the Frontend Dev Server](#7-run-the-frontend-dev-server)
- [8. Verify the Full Stack](#8-verify-the-full-stack)
- [9. Production Build (Frontend)](#9-production-build-frontend)
- [10. Common Issues & Troubleshooting](#10-common-issues--troubleshooting)
- [Appendix: Configuration Reference](#appendix-configuration-reference)

---

## 1. Prerequisites

| Software | Minimum Version | Download |
|---|---|---|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Ollama | latest | [ollama.com](https://ollama.com/download) |
| Git | any | [git-scm.com](https://git-scm.com/) |

**Check your existing installations:**

```bash
python --version      # Must be >= 3.10
node --version        # Must be >= 18
npm --version
ollama --version      # (skip if not installed yet)
git --version
```

---

## 2. Clone the Repository

```bash
git clone <repository-url>
cd ChatBox_UniCA
```

---

## 3. Setup Ollama (LLM Backend)

Ollama runs the large language models locally. The backend server auto-starts Ollama if it finds the binary, but starting it manually first is recommended for clarity.

### 3.1 Install Ollama

**macOS / Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from [ollama.com/download](https://ollama.com/download) and run it.

### 3.2 Pull the Required Model

The project uses `qwen2.5:32b` by default. Pull it:

```bash
ollama pull qwen2.5:32b
```

> This model is ~19 GB and may take a while to download. You can also use smaller models like `gemma3:4b` — see [Appendix: Changing the Model](#changing-the-model).

### 3.3 Start Ollama (if not already running)

```bash
ollama serve
```

This runs Ollama in the foreground. Open a new terminal for the next steps, or run it as a background service.

Verify it's ready:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see a JSON response listing your pulled models.

---

## 4. Setup Python Backend

### 4.1 Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should now show `(venv)`.

### 4.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you need PDF document support, also install:

```bash
pip install docling
```

> **Note about torch / CUDA:** The `sentence-transformers` package will install torch automatically. If you have an NVIDIA GPU and want GPU acceleration, install CUDA-enabled PyTorch *before* the requirements:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### 4.3 Build the Document Index

The project comes with a knowledge base at `docs/chroma/master.md`. Build the search index:

```bash
python build_index.py --doc-file ./docs/chroma/master.md
```

Expected output (approximate):

```
[INFO] Building index from docs/chroma/master.md ...
[INFO] Index built: X chunks in ./index_store
```

> This creates the `index_store/` directory containing Chroma vector store and BM25 index files.

---

## 5. Setup Vue Frontend

### 5.1 Install Node.js Dependencies

```bash
cd frontend
npm install
cd ..
```

### 5.2 Verify Frontend Environment

Check `frontend/.env` — it should contain:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The dev server also proxies `/api` requests to the backend (configured in `vite.config.js`), so this environment variable is optional when using the dev server. You can leave it as-is.

---

## 6. Run the Backend Server

### 6.1 Start the FastAPI Server

Make sure your virtual environment is activated and Ollama is running, then:

```bash
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

> **Important:** Do NOT use `uvicorn --factory api.main:create_app`. This project has a known CUDA/torch fork-safety issue that causes segmentation faults with `--factory`. Always use the `python -c` method above.

**What happens on startup:**

1. The server checks if Ollama is reachable at `http://127.0.0.1:11434`
2. If not reachable, it tries to auto-start Ollama from `~/ollama/bin/ollama` or PATH
3. If `OLLAMA_MODEL` env var is set, it auto-pulls that model
4. The RAG pipeline initializes (retriever, generator, cache, memory)

Successful startup log:

```
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Starting ChatBox_UniCA API server...
INFO: Ollama already reachable at http://127.0.0.1:11434
INFO: AppState initialized: retriever=True, generator=True, cache=False
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 6.2 Verify the Backend

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"ChatBox_UniCA"}
```

Also visit `http://127.0.0.1:8000/docs` in your browser — you should see the interactive Swagger API documentation.

---

## 7. Run the Frontend Dev Server

Open a **new terminal** (keep the backend running):

```bash
cd frontend
npm run dev
```

Output:

```
VITE v6.x.x  ready in XXX ms
➜  Local:   http://localhost:5173/
➜  Network: http://0.0.0.0:5173/
```

Open `http://localhost:5173` in your browser.

---

## 8. Verify the Full Stack

1. **Backend:** `http://127.0.0.1:8000/health` → `{"status":"ok","service":"ChatBox_UniCA"}`
2. **Frontend:** `http://localhost:5173` → ChatBox_UniCA UI should load
3. **Swagger Docs:** `http://127.0.0.1:8000/docs` → API docs
4. **Chat:** Type a question in the frontend (e.g., "What are the M1 prerequisites?") and verify the AI responds with relevant answers

If you see "Index not ready" in the frontend status panel, click the **Rebuild Index** button in the sidebar or rebuild manually:

```bash
python build_index.py --doc-file ./docs/chroma/master.md
```

---

## 9. Production Build (Frontend)

For serving the frontend via a static file server (e.g., nginx), build it:

```bash
cd frontend
npm run build
```

This generates a `frontend/dist/` directory. Serve it with any HTTP server:

```bash
# Option A: Use Vite's built-in preview server
npm run preview

# Option B: Use Python's http.server
python -m http.server 3000 -d dist

# Option C: Use nginx — point root to the dist/ directory
```

> If the frontend will access the backend at a different URL than where it was served from (e.g., different port or host), set `VITE_API_BASE_URL` in `frontend/.env` **before building**:
> ```
> VITE_API_BASE_URL=http://192.168.1.100:8000
> ```
> Then re-run `npm run build`.

---

## 10. Common Issues & Troubleshooting

### Ollama not reachable

**Symptom:** `OLLAMA_HOST` connection refused or startup hangs.

**Fix:**

```bash
# 1. Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# 2. If not, start it manually
ollama serve

# 3. Set custom host if Ollama is on a different machine
export OLLAMA_HOST=http://<remote-ip>:11434
```

### Index not found / "Index not ready"

**Symptom:** Backend reports index not ready; frontend shows "Index not ready."

**Fix:** Ensure the index is built:

```bash
python build_index.py --doc-file ./docs/chroma/master.md --persist-dir ./index_store
```

Then restart the backend server.

### Segmentation fault on startup (CUDA / torch fork issue)

**Symptom:** Backend crashes with segfault when using `uvicorn --factory`.

**Fix:** Always use the `python -c` method:

```bash
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)"
```

### Out of memory

**Symptom:** Ollama crashes or generation is extremely slow.

**Fix:**

- Use a smaller model: change `model_name` in `config.yaml` to `gemma3:4b` or `qwen2.5:7b`
- Reduce `rerank_candidates` in `config.yaml` from `30` to `10`
- Disable query expansion: set `query_expansion.enabled: false` in `config.yaml`

### Model not found

**Symptom:** Backend logs show model not found errors.

**Fix:** Pull the model manually:

```bash
ollama pull qwen2.5:32b
```

Or change the model in `config.yaml` to one you already have:

```yaml
generation:
  model_name: "gemma3:4b"
```

### CUDA out of memory (GPU users)

**Symptom:** `torch.cuda.OutOfMemoryError`.

**Fix:**

```bash
# Restrict Ollama to a specific GPU
export CUDA_VISIBLE_DEVICES=0
```

Or reduce batch sizes in `config.yaml` (not directly configurable; use a smaller model instead).

### "Address already in use" when starting the backend

**Fix:** Kill the existing process or use a different port:

```bash
# Find the process using port 8000
lsof -i :8000
kill <PID>

# Or use a different port
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
```

If you change the backend port, also update the frontend proxy in `frontend/vite.config.js` and `frontend/.env`.

### Frontend shows blank page or network errors

**Fix:**

1. Make sure the backend is running on port 8000
2. Check the browser's Developer Console (F12 → Console) for CORS or network errors
3. If the backend is on a different port, update `VITE_API_BASE_URL` in `frontend/.env` and restart the dev server

---

## Appendix: Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_BIN_DIR` | `~/ollama/bin` | Where to find the `ollama` binary |
| `OLLAMA_LOG_FILE` | `~/ollama/ollama.log` | Ollama log file path |
| `OLLAMA_MODEL` | *(not set)* | Model to auto-pull on server startup |
| `OLLAMA_STARTUP_WAIT` | `30` | Seconds to wait for Ollama to become ready |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device for Ollama (set empty to use CPU) |

### Key Config File (`config.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `indexing` | `persist_dir` | `./index_store` | Index storage directory |
| `indexing` | `force_rebuild` | `false` | Rebuild index on every startup |
| `generation` | `model_name` | `qwen2.5:32b` | LLM for answer generation |
| `generation` | `temperature` | `0.0` | Generation temperature (0 = deterministic) |
| `generation` | `num_predict` | `256` | Max output tokens |
| `retrieval` | `weight_vec` | `1.25` | Vector search weight in hybrid retrieval |
| `retrieval` | `weight_bm25` | `0.75` | BM25 keyword weight in hybrid retrieval |
| `cache` | `enabled` | `false` | Enable semantic cache |
| `query_expansion` | `enabled` | `true` | Enable query expansion (paraphrase + translation) |
| `memory` | `enabled` | `true` | Enable conversation memory |

### Changing the Model

To use a different LLM, edit `config.yaml`:

```yaml
generation:
  model_name: "gemma3:4b"        # Change this
  temperature: 0.0

query_expansion:
  model_name: "gemma3:4b"        # Change this too (if expansion is enabled)
```

Then pull the new model:

```bash
ollama pull gemma3:4b
```

Restart the backend server for changes to take effect.

### Changing the Backend Port

```bash
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8080)"
```

If you change the port, update:

1. `frontend/.env`: `VITE_API_BASE_URL=http://127.0.0.1:8080`
2. `frontend/vite.config.js`: proxy target → `http://127.0.0.1:8080`
3. Restart the frontend dev server

### API Endpoints Quick Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/` | API info and endpoint listing |
| `POST` | `/api/v1/chat/` | Chat (JSON response) |
| `POST` | `/api/v1/chat/stream` | Chat (SSE streaming) |
| `POST` | `/api/v1/sessions/new` | Create a new chat session |
| `DELETE` | `/api/v1/sessions/{id}` | Clear a session |
| `GET` | `/api/v1/index/status` | Index status |
| `POST` | `/api/v1/index/rebuild` | Rebuild the index |
| `GET` | `/api/v1/config/` | View current config |
| `PUT` | `/api/v1/config/` | Update config at runtime |

---

### Quick Start (TL;DR)

```bash
# 1. Install Ollama and pull the model
ollama pull qwen2.5:32b
ollama serve &
# (wait a few seconds)

# 2. Setup Python backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install docling             # optional, for PDF support
python build_index.py --doc-file ./docs/chroma/master.md

# 3. Start the backend API (keep this terminal running)
python -c "from api.main import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"

# 4. In a new terminal, start the frontend
cd frontend
npm install
npm run dev

# 5. Open http://localhost:5173 in your browser
```