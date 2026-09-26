"""Thin client around the Ollama HTTP API for embeddings and generation.

Everything runs locally through Ollama so the corpus never leaves the machine.
"""
from __future__ import annotations

from typing import List

import httpx

from . import config


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _client() -> httpx.Client:
    return httpx.Client(base_url=config.OLLAMA_HOST, timeout=config.REQUEST_TIMEOUT)


def ping() -> dict:
    """Return Ollama version info; raises OllamaError if the server is down."""
    try:
        with _client() as c:
            r = c.get("/api/version")
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the API
        raise OllamaError(f"Cannot reach Ollama at {config.OLLAMA_HOST}: {exc}") from exc


def list_models() -> List[str]:
    try:
        with _client() as c:
            r = c.get("/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Failed to list Ollama models: {exc}") from exc


def embed(
    texts: List[str], model: str | None = None, keep_alive: str = "30m"
) -> List[List[float]]:
    """Embed a batch of texts. Uses the /api/embed endpoint (batched).

    `keep_alive` keeps the embed model resident between calls, which removes the
    dominant per-request model-load overhead during a full ingest.
    """
    model = model or config.EMBED_MODEL
    if not texts:
        return []
    try:
        with _client() as c:
            r = c.post(
                "/api/embed",
                json={"model": model, "input": texts, "keep_alive": keep_alive},
            )
            r.raise_for_status()
            data = r.json()
            return data["embeddings"]
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Embedding request failed ({model}): {exc}") from exc


def generate(prompt: str, system: str | None = None, model: str | None = None) -> str:
    """Single-shot generation (non-streaming) for grounded answer synthesis."""
    model = model or config.LLM_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",  # keep the LLM resident between /ask calls
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.LLM_NUM_CTX,
        },
    }
    if system:
        payload["system"] = system
    try:
        with _client() as c:
            r = c.post("/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Generation request failed ({model}): {exc}") from exc
