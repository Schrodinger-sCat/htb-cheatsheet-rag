"""Index build / persist / load — embedding (dense) retrieval only.

Dense vectors live in a **ChromaDB** persistent collection at `data/index/chroma/`.
Chroma owns the vectors, their metadata, and the approximate-nearest-neighbour
search (cosine space). The chunk corpus + metadata is also written to
`chunks.jsonl` so we can map Chroma's returned ids back to full Chunk objects and
compute corpus statistics.

On disk in `data/index/`:
    chunks.jsonl   - one JSON chunk per line (corpus + metadata)
    chroma/        - ChromaDB persistent store (dense vectors)
    meta.json      - build info (models, counts)
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config, llm
from .ingest import Chunk, build_chunks

COLLECTION_NAME = "htb_writeups"


def _chroma_dir(index_dir: Path) -> Path:
    return index_dir / "chroma"


def _chroma_client(index_dir: Path):
    """Return a Chroma client.

    Standalone server if RAG_CHROMA_HOST is set (HttpClient), otherwise an
    embedded, on-disk store (PersistentClient) under `index_dir/chroma`.
    """
    import chromadb
    from chromadb.config import Settings

    if config.CHROMA_HOST:
        return chromadb.HttpClient(
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            ssl=config.CHROMA_SSL,
            settings=Settings(anonymized_telemetry=False),
        )
    return chromadb.PersistentClient(
        path=str(_chroma_dir(index_dir)),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


@dataclass
class RagIndex:
    chunks: List[Chunk]
    collection: object              # Chroma collection (None only if not built)
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


def _embed_batch_resilient(texts: List[str]) -> List[List[float]]:
    """Embed a batch, splitting and retrying on failure down to one item.

    Large batches are fast but can fail on a constrained machine (low disk/memory,
    an oversized input, a transient Ollama error). Rather than abort the whole
    ingest, we halve the batch and retry, isolating any single item that truly
    cannot be embedded and re-raising only then.
    """
    try:
        return llm.embed(texts)
    except Exception:
        if len(texts) <= 1:
            raise  # a single chunk that genuinely can't be embedded
        mid = len(texts) // 2
        return _embed_batch_resilient(texts[:mid]) + _embed_batch_resilient(texts[mid:])


def _embed_all(
    chunks: List[Chunk],
    batch_size: int | None = None,
    workers: int | None = None,
    progress=None,
) -> List[List[float]]:
    """Embed every chunk with concurrent batches.

    Ollama's per-request model-load cost dominates small batches, so we default to
    big ones; a failed batch is split and retried (see `_embed_batch_resilient`)
    so ingestion never hard-fails on a single oversized or transient case.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    batch_size = batch_size or config.EMBED_BATCH_SIZE
    workers = workers or config.EMBED_WORKERS

    texts = [c.retrieval_text() for c in chunks]
    batches = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: dict[int, List[List[float]]] = {}
    done = {"n": 0}
    lock = threading.Lock()

    def work(item):
        start, batch = item
        emb = _embed_batch_resilient(batch)
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


def _hnsw_metadata() -> dict:
    return {
        "hnsw:space": "cosine",
        "hnsw:M": config.CHROMA_HNSW_M,
        "hnsw:construction_ef": config.CHROMA_HNSW_CONSTRUCTION_EF,
        "hnsw:search_ef": config.CHROMA_HNSW_SEARCH_EF,
    }


def _populate_chroma(client, chunks: List[Chunk], embeddings: List[List[float]]):
    """(Re)create the Chroma collection and add all chunk vectors.

    Tuned HNSW params (see config) are essential: Chroma's defaults produce a
    low-quality, non-deterministic graph on a corpus this size, so search returns
    near-random hits on many builds. For the embedded store the directory is wiped
    in `build()` first; for a remote server we drop the collection here.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
        pass
    collection = client.create_collection(name=COLLECTION_NAME, metadata=_hnsw_metadata())
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


def build(raw_dir: Path | None = None, progress=None) -> RagIndex:
    index_dir = config.INDEX_DIR
    chunks = build_chunks(raw_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from {raw_dir or config.RAW_DIR}")
    id2idx = {c.id: i for i, c in enumerate(chunks)}

    # Embedded store: wipe the directory for a genuinely clean rebuild, and clear
    # Chroma's cached client (a PersistentClient is a per-path singleton, so an
    # in-process rebuild — e.g. POST /ingest — would otherwise reuse stale state).
    if not config.CHROMA_HOST:
        shutil.rmtree(_chroma_dir(index_dir), ignore_errors=True)
        _clear_chroma_cache()

    embeddings = _embed_all(chunks, progress=progress)
    client = _chroma_client(index_dir)
    collection = _populate_chroma(client, chunks, embeddings)

    index = RagIndex(chunks, collection, id2idx, config.EMBED_MODEL, config.LLM_MODEL)
    _verify_index(index)
    return index


def _clear_chroma_cache() -> None:
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _verify_index(index: RagIndex, samples: int = 8) -> None:
    """Sanity-check HNSW recall: a stored vector must find itself as nearest.

    A bad HNSW build (see _populate_chroma) returns near-random neighbours; this
    catches it at build time instead of letting the app serve silent garbage.
    """
    if index.collection is None or index.size == 0:
        return
    step = max(1, index.size // samples)
    ids = [index.chunks[i].id for i in range(0, index.size, step)][:samples]
    got = index.collection.get(ids=ids, include=["embeddings"])
    misses = 0
    for cid, vec in zip(got["ids"], got["embeddings"]):
        res = index.collection.query(query_embeddings=[vec], n_results=1, include=["distances"])
        if not res["ids"][0] or res["ids"][0][0] != cid or res["distances"][0][0] > 0.05:
            misses += 1
    if misses:
        raise RuntimeError(
            f"ChromaDB HNSW self-check failed ({misses}/{len(ids)} probes did not "
            "return themselves) — the vector index is unreliable. Try raising "
            "RAG_CHROMA_CONSTRUCTION_EF / RAG_CHROMA_HNSW_M and rebuilding."
        )


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
    id2idx = {c.id: i for i, c in enumerate(chunks)}

    collection = None
    embed_model = None
    meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").exists() else {}
    # Connect to a remote Chroma server whenever configured; otherwise only if
    # an embedded store exists on disk.
    if config.CHROMA_HOST or _chroma_dir(index_dir).exists():
        try:
            client = _chroma_client(index_dir)
            coll = client.get_collection(COLLECTION_NAME)
            if coll.count() > 0:
                collection = coll
                embed_model = meta.get("embed_model", config.EMBED_MODEL)
        except Exception:  # noqa: BLE001 - collection missing/empty/unreachable
            collection = None

    return RagIndex(chunks, collection, id2idx, embed_model, config.LLM_MODEL)


def exists(index_dir: Path | None = None) -> bool:
    index_dir = index_dir or config.INDEX_DIR
    return (index_dir / "chunks.jsonl").exists()
