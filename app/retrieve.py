"""Embedding (dense) retrieval.

The query is embedded with the same model the index was built with, and ChromaDB
returns the nearest passages by cosine similarity. This is a pure vector search:
no lexical/keyword component. See docs/design_note.md for why embedding-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import config, llm
from .index import RagIndex
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float          # cosine similarity in [-1, 1] (higher = closer)
    rank: int             # 0-based position in the result list

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "id": self.chunk.id,
            "machine": self.chunk.machine_title,
            "os": self.chunk.os,
            "difficulty": self.chunk.difficulty,
            "heading_path": self.chunk.heading_path,
            "source": self.chunk.source,
            "score": round(self.score, 6),
            "rank": self.rank,
        }
        if include_text:
            d["text"] = self.chunk.text
        return d


def search(index: RagIndex, query: str, top_k: int | None = None) -> List[Hit]:
    """Return the top-k passages nearest to the query in embedding space."""
    top_k = top_k or config.DEFAULT_TOP_K
    if index.collection is None:
        raise RuntimeError(
            "Dense index is not available. Rebuild with POST /ingest "
            "(embedding retrieval requires the ChromaDB vector store)."
        )

    # Embed the query with the SAME model the index was built with, then let
    # ChromaDB do the nearest-neighbour search (cosine space).
    q = llm.embed([query], model=index.embed_model)[0]
    res = index.collection.query(
        query_embeddings=[q],
        n_results=min(top_k, index.size),
        include=["distances"],
    )
    ids = res["ids"][0]
    distances = res["distances"][0]

    hits: List[Hit] = []
    for rank, (cid, dist) in enumerate(zip(ids, distances)):
        idx = index.id2idx.get(cid)
        if idx is None:
            continue
        # Chroma cosine "distance" is 1 - cosine_similarity.
        hits.append(Hit(chunk=index.chunks[idx], score=1.0 - dist, rank=rank))
    return hits
