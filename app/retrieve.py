"""Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with RRF.

Why hybrid (see docs/design_note.md): offensive-security queries mix exact
tokens the writers use verbatim ("GetUserSPNs", "SeImpersonatePrivilege",
"ADCS ESC1") with paraphrasable intent ("windows privilege escalation
cheatsheet"). BM25 nails the former, embeddings the latter. Reciprocal Rank
Fusion combines their rankings without needing to calibrate score scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import config, llm
from .index import RagIndex, tokenize
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical_rank: Optional[int]
    dense_rank: Optional[int]

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "id": self.chunk.id,
            "machine": self.chunk.machine_title,
            "os": self.chunk.os,
            "difficulty": self.chunk.difficulty,
            "heading_path": self.chunk.heading_path,
            "source": self.chunk.source,
            "score": round(self.score, 6),
            "lexical_rank": self.lexical_rank,
            "dense_rank": self.dense_rank,
        }
        if include_text:
            d["text"] = self.chunk.text
        return d


def _bm25_ranking(index: RagIndex, query: str, pool: int) -> List[int]:
    scores = index.bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:pool] if scores[i] > 0]


def _dense_ranking(index: RagIndex, query: str, pool: int) -> List[int]:
    if index.embeddings is None:
        return []
    # Embed the query with the SAME model the index was built with.
    q = np.asarray(llm.embed([query], model=index.embed_model)[0], dtype=np.float32)
    n = np.linalg.norm(q)
    if n:
        q = q / n
    sims = index.embeddings @ q
    order = np.argsort(sims)[::-1]
    return [int(i) for i in order[:pool]]


def search(
    index: RagIndex,
    query: str,
    top_k: int | None = None,
    use_dense: bool | None = None,
) -> List[Hit]:
    top_k = top_k or config.DEFAULT_TOP_K
    pool = max(config.CANDIDATE_POOL, top_k)
    use_dense = (index.embeddings is not None) if use_dense is None else (
        use_dense and index.embeddings is not None
    )

    lexical = _bm25_ranking(index, query, pool)
    dense = _dense_ranking(index, query, pool) if use_dense else []

    lex_rank = {idx: r for r, idx in enumerate(lexical)}
    dense_rank = {idx: r for r, idx in enumerate(dense)}

    # Reciprocal Rank Fusion.
    fused: dict[int, float] = {}
    for idx, r in lex_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + r + 1)
    for idx, r in dense_rank.items():
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + r + 1)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    hits: List[Hit] = []
    for idx, score in ranked:
        hits.append(
            Hit(
                chunk=index.chunks[idx],
                score=score,
                lexical_rank=lex_rank.get(idx),
                dense_rank=dense_rank.get(idx),
            )
        )
    return hits
