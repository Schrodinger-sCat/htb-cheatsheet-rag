#!/usr/bin/env python
"""Build and persist the RAG index from the command line.

Usage:
    python -m scripts.ingest            # hybrid (BM25 + dense embeddings)
    python -m scripts.ingest --no-dense # lexical only (no Ollama needed)
"""
import argparse
import sys
import time

from app import index as index_mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the HTB RAG index.")
    ap.add_argument("--no-dense", action="store_true", help="Skip embeddings (BM25 only).")
    args = ap.parse_args()

    start = time.time()
    last = {"t": 0.0}

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last["t"] > 2 or done == total:
            pct = 100 * done / total if total else 0
            print(f"  embedding {done}/{total} ({pct:.0f}%)", flush=True)
            last["t"] = now

    print("Building index...", flush=True)
    idx = index_mod.build(use_dense=not args.no_dense, progress=progress)
    index_mod.save(idx)
    print(f"Done in {time.time()-start:.1f}s", flush=True)
    print(idx.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
