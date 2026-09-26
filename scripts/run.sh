#!/usr/bin/env bash
# Start the HTB Cheatsheet Assistant API.
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv if present.
[ -d .venv ] && source .venv/bin/activate

export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
