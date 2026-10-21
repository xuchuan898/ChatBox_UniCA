# ChatBox_UniCA - Albert API edition

Branch: `develop_api_albert`

RAG chatbot for Universite Cote d'Azur CS Master questions. This branch uses
Albert API for answer generation, document reranking, and query rewriting by
default. It provides a command-line chatbot, a FastAPI backend, and a Vue 3 chat UI.

The original project documentation, including Ollama and remote GPU/tunnel setup,
is preserved in [README_OLLAMA.md](README_OLLAMA.md). Follow the instructions below
for this branch; its provider defaults differ from the original version.

## How it works

1. Load and split source documents, then persist a Chroma vector index and BM25 index.
2. Optionally rewrite/translate the question using Albert, including conversation history.
3. Retrieve candidate chunks locally using embeddings and BM25.
4. Send the question and candidate chunks to Albert's reranking endpoint.
5. Select the highest-ranked context and ask Albert to generate a grounded answer.

| Stage | Default model requested | Runs on |
| --- | --- | --- |
| Document/question embeddings | `intfloat/multilingual-e5-base` | Your machine/server |
| Candidate search | Chroma + BM25 | Your machine/server |
| Reranking | `openweight-rerank` | Albert API |
| Answer generation | `openweight-medium` | Albert API |
| Question rewriting/translation | `openweight-medium` | Albert API |

Albert model names are aliases and availability depends on your account. See the
[Albert API reference](https://albert.api.etalab.gouv.fr/reference) and authenticated
`GET /v1/models` for available models. No Ollama server or Ollama model pull is
needed with this branch's default configuration. Local embeddings still download
model weights on first use; a GPU is optional for that step. Questions and selected
chunk text are sent to Albert for inference.

## 1. Get the branch and install dependencies

Use Python 3.10+ (Python 3.11 recommended) and Git. Node.js/npm are needed only for
the optional frontend.

```bash
git clone --branch develop_api_albert https://github.com/xuchuan898/ChatBox_UniCA.git
cd ChatBox_UniCA
python -m venv .venv
```

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Albert uses Python's standard HTTP library; no additional Albert SDK is required.
Ollama libraries remain dependencies to support the alternative provider. PDF
conversion with Docling is optional; install `docling` if your input requires it.
Run all backend commands from the repository root so `config.yaml` and document
paths resolve correctly.

## 2. Configure your Albert API key

The key is not included in this repository. Choose one method:

**Environment variable (PowerShell):**

```powershell
$env:ALBERT_API_KEY = "YOUR_ALBERT_API_KEY"
```

**Environment variable (Linux/macOS):**

```bash
export ALBERT_API_KEY="YOUR_ALBERT_API_KEY"
```

**Private local file:** create `api_albert.txt` in the repository root, containing
only your API key. This file is explicitly ignored by Git. The default config
points to it, and relative key file paths resolve from the repository directory.

An environment variable takes priority over the file. Set the environment variable
in each terminal used to launch a backend, or use the local file for both CLI and
API startup. A `.env` file is not loaded automatically. Keep the key on the backend;
never put it in a `VITE_` frontend setting.

## 3. Launch the command-line chatbot

Ask one question:

```bash
python chat_box.py --doc-file ./docs/chroma/master.md -q "What are M1 prerequisites?"
```

Start an interactive conversation:

```bash
python chat_box.py --doc-file ./docs/chroma/master.md
```

Type `exit` to quit. Missing or stale indexes are built automatically. Optionally
build the index in advance:

```bash
python build_index.py --doc-file ./docs/chroma/master.md --persist-dir ./index_store
```

Process a question file, using your own existing input path:

```bash
python chat_box.py --question-file ./my_questions.txt --answer-file ./answers.txt
```

Use one question per line ending in `?`. `--debug` prints retrieval details and
prompts, so use it when you need to inspect the context sent to the model.

## 4. Launch the API server

In the activated Python environment with the key configured:

```bash
python -c "from api.main import create_app; import uvicorn; uvicorn.run(create_app(), host='127.0.0.1', port=8000)"
```

Use one worker to avoid loading multiple copies of the embedding model. The API
loads/builds the index on startup and warms up retrieval. Initialization may take
longer on first use because embeddings need to download. Ollama startup and model
pulling are skipped when generation and enabled query rewriting use Albert.

- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

Example request in PowerShell:

```powershell
$body = @{query = "What are M1 prerequisites?"; session_id = "demo"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/chat/" -ContentType "application/json" -Body $body
```

Example request on Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are M1 prerequisites?","session_id":"demo"}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/chat/` | Answer, sources, session ID, and elapsed time |
| `POST /api/v1/chat/stream` | SSE progress events followed by the final answer |
| `GET /api/v1/sessions/list` | List sessions |
| `POST /api/v1/sessions/new` | Create a session |
| `GET /api/v1/documents/list` | List available documents |
| `POST /api/v1/documents/select` | Select documents and build/load their index |
| `GET /api/v1/index/status` | Inspect the index |
| `GET /api/v1/config/` | Inspect sanitized configuration |

The SSE endpoint reports pipeline stages; it does not stream individual Albert
generation tokens. See `/docs` for all endpoints and schemas.

## 5. Launch the optional web interface

Keep the backend running. In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` requests to
`http://127.0.0.1:8000`. For a different backend address, set `VITE_API_BASE_URL`
according to [frontend/README.md](frontend/README.md). The frontend never needs
the Albert key. For production frontend assets, run `npm run build`.

## Model selection and configuration

The relevant defaults in `config.yaml` are:

```yaml
albert:
  base_url: https://albert.api.etalab.gouv.fr/v1
  api_key_env: ALBERT_API_KEY
  api_key_file: api_albert.txt
  timeout: 120

retrieval:
  rerank_provider: albert
  albert_rerank_model_name: openweight-rerank

query_expansion:
  enabled: true
  rewriter: albert
  albert_model_name: openweight-medium

generation:
  provider: albert
  albert_model_name: openweight-medium
  temperature: 0.0
  num_predict: 256
```

These are excerpts: edit the existing sections, preserving their other settings.
`num_predict` becomes `max_completion_tokens` in Albert generation requests.
`ALBERT_BASE_URL` overrides `albert.base_url`; a root URL or a URL ending in `/v1`
is accepted. `timeout` is the per-request timeout in seconds.

CLI arguments override the configuration for that run. For example, choose a
different generation alias:

```bash
python chat_box.py --albert-generation-model openweight-large -q "What are M1 prerequisites?"
```

Other model switches are `--albert-rerank-model` and `--albert-expansion-model`.
Only choose models supporting the corresponding task and available to your account.

To force Albert explicitly even with a different config file:

```bash
python chat_box.py --generation-provider albert --rerank-provider albert --expansion-rewriter albert -q "What are M1 prerequisites?"
```

You can mix providers. For example, use `--rerank-provider local` to load the local
cross-encoder specified by `retrieval.rerank_model_name`. Switch generation and
rewriting back with `--generation-provider ollama --expansion-rewriter ollama`;
those stages use their `model_name` values and require a running Ollama server
with those models installed. See [README_OLLAMA.md](README_OLLAMA.md).

For the API, update `config.yaml` and restart the server after changing providers,
model IDs, or credentials. Runtime parameter overrides do not rebuild model
instances. Semantic answer caching is disabled by default; clear cached answers
when changing models if you have enabled it. Existing experiment compatibility
functions and RAGAS scripts retain their original model behavior and are not
switched to Albert by this branch's runtime defaults.

## Tests and troubleshooting

Run the offline adapter tests without an API key or model downloads:

```bash
python -B -m unittest services.test_model_providers -v
```

The tests cover credential loading, HTTP request formatting, error handling,
document/score alignment, multi-query score pooling, and provider selection.
They mock Albert responses rather than making live inference requests.

- **Missing key:** set `ALBERT_API_KEY` or create the private `api_albert.txt` file.
- **HTTP 401/403:** check the key and account access.
- **HTTP 404:** check the base URL and requested model ID.
- **HTTP 429/503:** check quota or service availability and retry later.
- **Timeout or invalid JSON:** check connectivity, the configured endpoint, and
  any proxy that might return an HTML page instead of an API response.
- **Missing Python module:** activate the intended environment and install
  `requirements.txt` with that environment's `python -m pip`.
- **Slow first start / `model.safetensors` download:** local embeddings require
  Hugging Face weights even when inference uses Albert.
- **Changed documents:** use `--force-rebuild` in the CLI or rebuild/select
  documents through the API.

## Relevant files

| File | Role |
| --- | --- |
| `services/model_providers.py` | Albert HTTP client and provider factories |
| `services/test_model_providers.py` | Offline adapter regression tests |
| `chat_box.py` | CLI orchestration and provider/model arguments |
| `api/dependencies.py` | Construct shared API models and retrieval components |
| `api/main.py` | API lifecycle and conditional Ollama startup |
| `api/routers/documents.py` | Keep query provider selection during document changes |
| `config.yaml` | Albert-first branch defaults; no credentials |
| `core/` | Existing indexing, retrieval, prompts, and memory implementations |
| `frontend/` | Vue chat interface |

Local keys, virtual environments, Python caches, downloaded weights, indexes,
`frontend/node_modules`, and frontend build output are excluded from Git.
