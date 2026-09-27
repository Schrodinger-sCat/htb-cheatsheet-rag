#!/usr/bin/env python
"""Build and persist the RAG index from the command line.

Embeds every chunk (via Ollama) and stores the vectors in ChromaDB.

Usage:
    python -m scripts.ingest
"""
import argparse
import sys
import time

from app import index as index_mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the HTB RAG index (embedding retrieval).")
    ap.parse_args()

    start = time.time()
    last = {"t": 0.0}

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last["t"] > 2 or done == total:
            pct = 100 * done / total if total else 0
            print(f"  embedding {done}/{total} ({pct:.0f}%)", flush=True)
            last["t"] = now

    print("Building index...", flush=True)
    idx = index_mod.build(progress=progress)
    index_mod.save(idx)
    print(f"Done in {time.time()-start:.1f}s", flush=True)
    print(idx.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
