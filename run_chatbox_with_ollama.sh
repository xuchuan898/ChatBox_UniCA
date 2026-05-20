#!/usr/bin/env bash
set -euo pipefail

# Wrapper: ensure Ollama is available before running chat_box.py.
# Usage:
#   bash run_chatbox_with_ollama.sh --doc-file ./docs/chroma/master.md -q "Your question"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------- configurable defaults --------
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:4b}"
OLLAMA_BIN_DIR="${OLLAMA_BIN_DIR:-$HOME/ollama/bin}"
OLLAMA_LOG_FILE="${OLLAMA_LOG_FILE:-$HOME/ollama/ollama.log}"
USER_AGENT_VALUE="${USER_AGENT_VALUE:-MyChatBot/1.0}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-20}"
# --------------------------------------

mkdir -p "$(dirname "$OLLAMA_LOG_FILE")"

export USER_AGENT="$USER_AGENT_VALUE"
export OLLAMA_HOST
export PATH="$OLLAMA_BIN_DIR:$PATH"

find_ollama_bin() {
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  if [ -x "$OLLAMA_BIN_DIR/ollama" ]; then
    echo "$OLLAMA_BIN_DIR/ollama"
    return 0
  fi
  return 1
}

wait_ollama_ready() {
  local wait_s="$1"
  local i
  for ((i=0; i<wait_s; i++)); do
    if curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

OLLAMA_BIN="$(find_ollama_bin || true)"
if [ -z "${OLLAMA_BIN:-}" ]; then
  echo "[ERROR] ollama not found."
  echo "        Expected in PATH or at: $OLLAMA_BIN_DIR/ollama"
  echo "        Please install ollama or set OLLAMA_BIN_DIR."
  exit 1
fi

if ! curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "[INFO] Ollama not reachable at $OLLAMA_HOST, starting background service..."
  nohup "$OLLAMA_BIN" serve >>"$OLLAMA_LOG_FILE" 2>&1 &
  if ! wait_ollama_ready "$STARTUP_WAIT_SECONDS"; then
    echo "[ERROR] Ollama failed to become ready within ${STARTUP_WAIT_SECONDS}s."
    echo "        Check log: $OLLAMA_LOG_FILE"
    exit 1
  fi
fi

echo "[INFO] Ollama is ready: $OLLAMA_HOST"
echo "[INFO] Ensuring model exists: $OLLAMA_MODEL"
"$OLLAMA_BIN" pull "$OLLAMA_MODEL" >/dev/null

echo "[INFO] Launching chat_box.py"
python chat_box.py "$@"
