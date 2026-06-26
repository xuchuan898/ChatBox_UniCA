"""FastAPI application entry point (App Factory pattern)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen, Request as URLRequest

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_app_state
from api.exceptions import IndexNotReadyError, GenerationTimeoutError
from api.routers import chat, config as config_router, index, sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_BIN_DIR = os.environ.get(
    "OLLAMA_BIN_DIR", str(Path.home() / "ollama" / "bin")
)
OLLAMA_LOG_FILE = os.environ.get(
    "OLLAMA_LOG_FILE", str(Path.home() / "ollama" / "ollama.log")
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", None)
STARTUP_WAIT_SECONDS = int(os.environ.get("OLLAMA_STARTUP_WAIT", "30"))


def _ollama_is_reachable() -> bool:
    try:
        req = URLRequest(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _find_ollama_bin() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidate = Path(OLLAMA_BIN_DIR) / "ollama"
    return str(candidate) if candidate.is_file() else None


def _ensure_ollama() -> None:
    if _ollama_is_reachable():
        logger.info("Ollama already reachable at %s", OLLAMA_HOST)
        return

    bin_path = _find_ollama_bin()
    if not bin_path:
        logger.warning(
            "Ollama binary not found in PATH or %s. "
            "Skipping auto-start. Start Ollama manually.",
            OLLAMA_BIN_DIR,
        )
        return

    log_path = Path(OLLAMA_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Ollama from %s ...", bin_path)
    with log_path.open("a") as log_fp:
        subprocess.Popen(
            [bin_path, "serve"],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )

    deadline = time.time() + STARTUP_WAIT_SECONDS
    while time.time() < deadline:
        if _ollama_is_reachable():
            logger.info("Ollama is ready at %s", OLLAMA_HOST)
            return
        time.sleep(1)

    logger.warning(
        "Ollama did not become ready within %ds. Check log: %s",
        STARTUP_WAIT_SECONDS,
        OLLAMA_LOG_FILE,
    )


def _ensure_model() -> None:
    model = OLLAMA_MODEL
    if not model:
        return
    bin_path = _find_ollama_bin()
    if not bin_path:
        return
    logger.info("Ensuring model %s is pulled ...", model)
    try:
        subprocess.run(
            [bin_path, "pull", model],
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("Failed to pull model %s: %s", model, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting ChatBox_UniCA API server...")
    _ensure_ollama()
    _ensure_model()
    state = get_app_state()
    logger.info(
        "AppState initialized: retriever=%s, generator=%s, cache=%s",
        bool(state.retriever),
        bool(state.generator),
        bool(state.cache),
    )
    # Warmup: trigger model loading (embedding, reranker) before first user request
    asyncio.ensure_future(_warmup_models(state))
    yield
    logger.info("Shutting down ChatBox_UniCA API server...")


async def _warmup_models(state) -> None:
    """Warm up embedding/reranker models so first user request is fast."""
    try:
        logger.info("Warming up models ...")
        docs = await asyncio.to_thread(state.retriever.retrieve, "warmup", top_n=1)
        if docs:
            await asyncio.to_thread(
                state.retriever.reranker.rerank, "warmup", docs[:1], top_n=1
            )
        logger.info("Warmup complete")
    except Exception:
        logger.warning("Warmup failed (non-fatal)", exc_info=True)


def create_app() -> FastAPI:
    """App Factory: create and return a FastAPI instance."""
    app = FastAPI(
        title="ChatBox_UniCA API",
        description="RAG-based Q&A API for Université Côte d'Azur CS Master",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(index.router)
    app.include_router(config_router.router)

    # Global exception handlers
    @app.exception_handler(IndexNotReadyError)
    async def index_not_ready_handler(request: Request, exc: IndexNotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GenerationTimeoutError)
    async def generation_timeout_handler(request: Request, exc: GenerationTimeoutError):
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
        )

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "ChatBox_UniCA"}

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "ChatBox_UniCA",
            "version": "1.0.0",
            "description": "RAG-based Q&A system for Université Côte d'Azur CS Master",
            "docs": "/docs",
            "endpoints": {
                "health": "GET /health",
                "chat": "POST /api/v1/chat/",
                "chat_stream": "POST /api/v1/chat/stream",
                "sessions_create": "POST /api/v1/sessions/new",
                "sessions_clear": "DELETE /api/v1/sessions/{session_id}",
                "index_status": "GET /api/v1/index/status",
                "index_rebuild": "POST /api/v1/index/rebuild",
                "index_update": "POST /api/v1/index/update",
                "config_get": "GET /api/v1/config/",
                "config_update": "PUT /api/v1/config/",
            },
        }

    return app