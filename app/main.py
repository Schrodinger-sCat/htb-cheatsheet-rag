"""FastAPI service: the HTB Cheatsheet Assistant (RAG).

Endpoints
  GET  /health          - liveness + Ollama reachability + index status
  GET  /stats           - corpus / index statistics
  POST /ingest          - (re)build the index from the raw write-ups
  GET  /search          - retrieval only (ranked passages, no LLM)
  POST /ask             - full RAG: retrieve + grounded, cited synthesis
"""
from __future__ import annotations

import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config, index as index_mod, llm
from .retrieve import search as retrieve_search
from .synthesize import synthesize, citations

app = FastAPI(
    title="HTB Cheatsheet Assistant (RAG)",
    description="Grounded, cited answers over the HTB write-up corpus via Ollama.",
    version="1.0.0",
)

# --- in-memory index state ------------------------------------------------
_index: Optional[index_mod.RagIndex] = None
_index_lock = threading.Lock()
_build_status = {"building": False, "processed": 0, "total": 0}


def _get_index() -> index_mod.RagIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                if not index_mod.exists():
                    raise HTTPException(
                        status_code=409,
                        detail="Index not built yet. Call POST /ingest first.",
                    )
                _index = index_mod.load()
    return _index


@app.on_event("startup")
def _startup() -> None:
    global _index
    if index_mod.exists():
        try:
            _index = index_mod.load()
        except Exception:  # noqa: BLE001 - lazy load will retry on demand
            _index = None


# --- schemas --------------------------------------------------------------
class IngestRequest(BaseModel):
    force: bool = Field(False, description="Rebuild even if an index already exists.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Provide me the Windows privilege escalation cheatsheet."])
    top_k: Optional[int] = Field(None, ge=1, le=30)
    include_context: bool = Field(False, description="Return the retrieved passages too.")


# --- endpoints ------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    ollama_ok, ollama_detail = True, None
    try:
        llm.ping()
    except Exception as exc:  # noqa: BLE001
        ollama_ok, ollama_detail = False, str(exc)
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "ollama_detail": ollama_detail,
        "index_built": index_mod.exists(),
        "index_loaded": _index is not None,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "building": _build_status["building"],
    }


@app.get("/stats")
def stats() -> dict:
    idx = _get_index()
    return idx.stats()


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _index
    if _build_status["building"]:
        raise HTTPException(status_code=409, detail="An ingest is already running.")
    if index_mod.exists() and not req.force:
        raise HTTPException(
            status_code=409,
            detail="Index already exists. Pass force=true to rebuild.",
        )

    def _progress(done: int, total: int) -> None:
        _build_status["processed"] = done
        _build_status["total"] = total

    _build_status.update({"building": True, "processed": 0, "total": 0})
    try:
        new_index = index_mod.build(progress=_progress)
        index_mod.save(new_index)
        with _index_lock:
            _index = new_index
    except llm.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    finally:
        _build_status["building"] = False
    return {"status": "built", **new_index.stats()}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Query text."),
    top_k: int = Query(config.DEFAULT_TOP_K, ge=1, le=30),
    include_text: bool = Query(True),
) -> dict:
    idx = _get_index()
    try:
        hits = retrieve_search(idx, q, top_k=top_k)
    except llm.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "query": q,
        "count": len(hits),
        "results": [h.to_dict(include_text=include_text) for h in hits],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    idx = _get_index()
    try:
        hits = retrieve_search(idx, req.question, top_k=req.top_k)
        answer = synthesize(req.question, hits)
    except llm.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    resp = {
        "question": req.question,
        "answer": answer,
        "citations": citations(hits),
        "retrieved": len(hits),
    }
    if req.include_context:
        resp["context"] = [h.to_dict(include_text=True) for h in hits]
    return resp


@app.get("/")
def root() -> dict:
    return {
        "service": "HTB Cheatsheet Assistant (RAG)",
        "docs": "/docs",
        "endpoints": ["/health", "/stats", "/ingest", "/search", "/ask"],
    }
