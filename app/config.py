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

# --- Ollama ---------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Synthesis model: "llama 3.2 latest 2gb" -> the 3B instruct build is ~2.0 GB.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2:latest")
# all-minilm (45 MB, 384-dim) embeds ~10x faster than nomic-embed-text on CPU
# and, paired with BM25 in the hybrid retriever, is plenty for paraphrase recall.
# Set RAG_EMBED_MODEL=nomic-embed-text for higher-quality (slower) embeddings.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-minilm")
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "180"))

# --- Chunking -------------------------------------------------------------
# Chunks are built from Markdown headings; long sections are split further.
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_CHARS = int(os.getenv("RAG_MIN_CHUNK_CHARS", "80"))

# --- Retrieval ------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = int(os.getenv("RAG_CANDIDATE_POOL", "40"))
# Reciprocal-rank-fusion constant.
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
# Set to "0" to disable dense retrieval (pure lexical / offline mode).
USE_DENSE = os.getenv("RAG_USE_DENSE", "1") == "1"

# --- Generation -----------------------------------------------------------
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
# 4096 comfortably fits top-k passages + prompt and is markedly faster on CPU
# than 8192; raise it if you increase RAG_TOP_K substantially.
LLM_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "4096"))
