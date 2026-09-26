"""Index build / persist / load.

The index has two parts that live side by side on disk in `data/index/`:
  * chunks.jsonl         - one JSON chunk per line (the corpus + metadata)
  * embeddings.npy       - float32 matrix [n_chunks, dim] aligned with chunks
  * meta.json            - build info (models, counts, dim)

BM25 is rebuilt in memory from chunks.jsonl on load (cheap), so only the dense
matrix needs to be persisted.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, llm
from .ingest import Chunk, build_chunks

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RagIndex:
    chunks: List[Chunk]
    bm25: BM25Okapi
    embeddings: Optional[np.ndarray]  # L2-normalized, or None if dense disabled
    embed_model: Optional[str]
    llm_model: str

    @property
    def size(self) -> int:
        return len(self.chunks)

    def stats(self) -> dict:
        machines = {c.machine for c in self.chunks}
        os_counts: dict[str, int] = {}
        for c in self.chunks:
            os_counts[c.os] = os_counts.get(c.os, 0) + 1
        return {
            "chunks": len(self.chunks),
            "machines": len(machines),
            "os_breakdown": os_counts,
            "dense_enabled": self.embeddings is not None,
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
        }


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _embed_all(
    chunks: List[Chunk],
    batch_size: int = 384,
    workers: int = 3,
    progress=None,
) -> np.ndarray:
    """Embed every chunk with concurrent large batches.

    Ollama's per-request model-load cost dominates small batches, so we use big
    batches; and the CPU embed model is under-utilized by a single stream, so we
    run a few batches concurrently. Results are written back in index order.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    texts = [c.retrieval_text() for c in chunks]
    batches = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: dict[int, List[List[float]]] = {}
    done = {"n": 0}
    lock = threading.Lock()

    def work(item):
        start, batch = item
        emb = llm.embed(batch)
        with lock:
            results[start] = emb
            done["n"] += len(batch)
            if progress:
                progress(done["n"], len(texts))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, batches))

    vecs: List[List[float]] = []
    for start, _ in batches:
        vecs.extend(results[start])
    return _normalize(np.asarray(vecs, dtype=np.float32))


def build(raw_dir: Path | None = None, use_dense: bool | None = None, progress=None) -> RagIndex:
    use_dense = config.USE_DENSE if use_dense is None else use_dense
    chunks = build_chunks(raw_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from {raw_dir or config.RAW_DIR}")
    bm25 = BM25Okapi([tokenize(c.retrieval_text()) for c in chunks])
    embeddings = None
    embed_model = None
    if use_dense:
        embeddings = _embed_all(chunks, progress=progress)
        embed_model = config.EMBED_MODEL
    return RagIndex(chunks, bm25, embeddings, embed_model, config.LLM_MODEL)


def save(index: RagIndex, index_dir: Path | None = None) -> None:
    index_dir = index_dir or config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in index.chunks:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    if index.embeddings is not None:
        np.save(index_dir / "embeddings.npy", index.embeddings)
    meta = {
        "built_at": time.time(),
        "chunks": index.size,
        "embed_model": index.embed_model,
        "llm_model": index.llm_model,
        "dense_enabled": index.embeddings is not None,
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load(index_dir: Path | None = None) -> RagIndex:
    index_dir = index_dir or config.INDEX_DIR
    chunks_file = index_dir / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"No index at {index_dir}. Run POST /ingest (or scripts/ingest.py) first."
        )
    chunks: List[Chunk] = []
    with chunks_file.open(encoding="utf-8") as fh:
        for line in fh:
            chunks.append(Chunk(**json.loads(line)))
    bm25 = BM25Okapi([tokenize(c.retrieval_text()) for c in chunks])
    embeddings = None
    embed_model = None
    emb_file = index_dir / "embeddings.npy"
    if emb_file.exists():
        embeddings = np.load(emb_file)
        meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").exists() else {}
        embed_model = meta.get("embed_model", config.EMBED_MODEL)
    return RagIndex(chunks, bm25, embeddings, embed_model, config.LLM_MODEL)


def exists(index_dir: Path | None = None) -> bool:
    index_dir = index_dir or config.INDEX_DIR
    return (index_dir / "chunks.jsonl").exists()
