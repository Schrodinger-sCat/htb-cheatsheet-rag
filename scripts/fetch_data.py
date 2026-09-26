#!/usr/bin/env python
"""Fetch the HTB write-up corpus into data/raw/.

The corpus is NOT vendored in this repo (the write-ups contain example CTF
tokens/keys from retired machines that trip secret scanners). This script pulls
the exact same files from the upstream source so the RAG is fully reproducible.

Source: https://github.com/0xh7ml/htb-wiki (raw/)

Usage:
    python -m scripts.fetch_data          # git clone (fast, needs git)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/0xh7ml/htb-wiki.git"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Cloning {REPO} ...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO, tmp],
            check=True,
        )
        src = Path(tmp) / "raw"
        if not src.is_dir():
            print("error: 'raw/' not found in upstream repo", file=sys.stderr)
            return 1
        count = 0
        for md in src.glob("*.md"):
            shutil.copy2(md, RAW_DIR / md.name)
            count += 1
    print(f"Fetched {count} write-ups into {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
