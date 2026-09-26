"""Index build / persist / load.

The index has two halves:

  * **Lexical (BM25)** — rebuilt in memory from `chunks.jsonl` on load. Cheap, so
    only the chunk corpus is persisted as JSONL (it also drives the pure-lexical,
    no-embeddings mode).
  * **Dense (embeddings)** — stored in a **ChromaDB** persistent collection at
    `data/index/chroma/`. Chroma owns the vectors, their metadata, and the
    approximate-nearest-neighbour search (cosine space).

On disk in `data/index/`:
    chunks.jsonl   - one JSON chunk per line (corpus + metadata; BM25 source)
    chroma/        - ChromaDB persistent store (dense vectors)
    meta.json      - build info (models, counts, dense flag)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from . import config, llm
from .ingest import Chunk, build_chunks

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
COLLECTION_NAME = "htb_writeups"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _chroma_dir(index_dir: Path) -> Path:
    return index_dir / "chroma"


def _chroma_client(index_dir: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(_chroma_dir(index_dir)),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


@dataclass
class RagIndex:
    chunks: List[Chunk]
    bm25: BM25Okapi
    collection: object              # Chroma collection, or None if dense disabled
    id2idx: dict                    # chunk.id -> position in `chunks`
    embed_model: Optional[str]
    llm_model: str

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dense_enabled(self) -> bool:
        return self.collection is not None

    def stats(self) -> dict:
        machines = {c.machine for c in self.chunks}
        os_counts: dict[str, int] = {}
        for c in self.chunks:
            os_counts[c.os] = os_counts.get(c.os, 0) + 1
        dim = None
        vectors = 0
        if self.collection is not None:
            vectors = self.collection.count()
            peek = self.collection.peek(limit=1)
            embs = peek.get("embeddings")
            if embs is not None and len(embs):
                dim = len(embs[0])
        return {
            "chunks": len(self.chunks),
            "machines": len(machines),
            "os_breakdown": os_counts,
            "dense_enabled": self.dense_enabled,
            "vector_store": "chromadb" if self.dense_enabled else None,
            "vectors": vectors,
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "embedding_dim": dim,
        }


def _embed_all(
    chunks: List[Chunk],
    batch_size: int = 384,
    workers: int = 3,
    progress=None,
) -> List[List[float]]:
    """Embed every chunk with concurrent large batches.

    Ollama's per-request model-load cost dominates small batches, so we use big
    batches. Results are written back in index order.
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
    return vecs


def _populate_chroma(client, chunks: List[Chunk], embeddings: List[List[float]]):
    """(Re)create the Chroma collection and add all chunk vectors."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    add_batch = 2000
    for i in range(0, len(chunks), add_batch):
        sl = slice(i, i + add_batch)
        batch = chunks[sl]
        collection.add(
            ids=[c.id for c in batch],
            embeddings=embeddings[sl],
            documents=[c.text for c in batch],
            metadatas=[
                {"machine": c.machine, "machine_title": c.machine_title, "os": c.os}
                for c in batch
            ],
        )
    return collection


def build(raw_dir: Path | None = None, use_dense: bool | None = None, progress=None) -> RagIndex:
    use_dense = config.USE_DENSE if use_dense is None else use_dense
    index_dir = config.INDEX_DIR
    chunks = build_chunks(raw_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from {raw_dir or config.RAW_DIR}")
    bm25 = BM25Okapi([tokenize(c.retrieval_text()) for c in chunks])
    id2idx = {c.id: i for i, c in enumerate(chunks)}

    collection = None
    embed_model = None
    if use_dense:
        embeddings = _embed_all(chunks, progress=progress)
        client = _chroma_client(index_dir)
        collection = _populate_chroma(client, chunks, embeddings)
        embed_model = config.EMBED_MODEL

    return RagIndex(chunks, bm25, collection, id2idx, embed_model, config.LLM_MODEL)


def save(index: RagIndex, index_dir: Path | None = None) -> None:
    index_dir = index_dir or config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in index.chunks:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    # Chroma persists itself; we only record build metadata here.
    meta = {
        "built_at": time.time(),
        "chunks": index.size,
        "embed_model": index.embed_model,
        "llm_model": index.llm_model,
        "dense_enabled": index.dense_enabled,
        "vector_store": "chromadb" if index.dense_enabled else None,
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
    id2idx = {c.id: i for i, c in enumerate(chunks)}

    collection = None
    embed_model = None
    meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").exists() else {}
    if _chroma_dir(index_dir).exists():
        try:
            client = _chroma_client(index_dir)
            coll = client.get_collection(COLLECTION_NAME)
            if coll.count() > 0:
                collection = coll
                embed_model = meta.get("embed_model", config.EMBED_MODEL)
        except Exception:  # noqa: BLE001 - no/empty collection -> lexical-only
            collection = None

    return RagIndex(chunks, bm25, collection, id2idx, embed_model, config.LLM_MODEL)


def exists(index_dir: Path | None = None) -> bool:
    index_dir = index_dir or config.INDEX_DIR
    return (index_dir / "chunks.jsonl").exists()
