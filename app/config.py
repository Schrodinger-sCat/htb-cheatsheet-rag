"""Central configuration for the HTB Cheatsheet Assistant RAG service.

Every value can be overridden with an environment variable so the same code runs
locally, in CI, or in a container without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.getenv("RAG_RAW_DIR", BASE_DIR / "data" / "raw"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "data" / "index"))

# --- ChromaDB -------------------------------------------------------------
# By default Chroma runs *embedded* (in-process) and persists to INDEX_DIR/chroma.
# To use a standalone Chroma server instead (e.g. `chroma run --host 0.0.0.0
# --port 8001`, or a Docker container), set RAG_CHROMA_HOST (and optionally
# RAG_CHROMA_PORT). When a host is set, the app connects with HttpClient and the
# vectors live on that server rather than on local disk.
CHROMA_HOST = os.getenv("RAG_CHROMA_HOST")  # e.g. "localhost"; unset = embedded
CHROMA_PORT = int(os.getenv("RAG_CHROMA_PORT", "8000"))
CHROMA_SSL = os.getenv("RAG_CHROMA_SSL", "0") == "1"

# --- Ollama ---------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Synthesis model: "llama 3.2 latest 2gb" -> the 3B instruct build is ~2.0 GB.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2:latest")
# all-minilm (45 MB, 384-dim) embeds ~10x faster than nomic-embed-text on CPU
# and is plenty accurate for the embedding retrieval here.
# Set RAG_EMBED_MODEL=nomic-embed-text for higher-quality (slower) embeddings.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-minilm")
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "180"))

# --- Chunking -------------------------------------------------------------
# Chunks are built from Markdown headings; long sections are split further.
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_CHARS = int(os.getenv("RAG_MIN_CHUNK_CHARS", "80"))

# --- Retrieval ------------------------------------------------------------
# Passages returned by the embedding search (and fed to the LLM).
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "8"))

# --- Generation -----------------------------------------------------------
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
# 4096 comfortably fits top-k passages + prompt and is markedly faster on CPU
# than 8192; raise it if you increase RAG_TOP_K substantially.
LLM_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "4096"))
