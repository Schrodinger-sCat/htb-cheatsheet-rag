"""Ingestion: turn the raw HTB Markdown write-ups into retrievable chunks.

Design choices (see docs/design_note.md):
  * One document == one HTB machine (filename `htb-<machine>.md`).
  * Chunks are heading-aware: we split on Markdown headings so each chunk keeps
    a semantic boundary (e.g. a single "### GetUserSPNs" step) and we carry the
    heading breadcrumb + machine name into the text so retrieval has context.
  * Over-long sections (big code/tool dumps) are windowed with overlap so we
    never feed a monster chunk to the embedder or the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List

from . import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
OS_RE = re.compile(r"!\[(Windows|Linux|FreeBSD|Android|Other)\]", re.IGNORECASE)
DIFFICULTY_RE = re.compile(r"^\s*(Easy|Medium|Hard|Insane)\s*$", re.IGNORECASE)
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@dataclass
class Chunk:
    id: str
    machine: str            # slug, e.g. "absolute"
    machine_title: str      # display, e.g. "Absolute"
    os: str                 # "Windows" / "Linux" / "Unknown"
    difficulty: str         # "Easy" / ... / "Unknown"
    heading_path: str       # breadcrumb, e.g. "Shell as svc / Kerberoast"
    text: str               # the chunk body
    source: str             # source filename

    def retrieval_text(self) -> str:
        """Text actually indexed/embedded: machine + heading give lexical and
        semantic anchors that the raw body often lacks."""
        header = f"[Machine: {self.machine_title} | OS: {self.os}] {self.heading_path}"
        return f"{header}\n{self.text}"


def machine_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem  # htb-absolute
    slug = re.sub(r"^htb-", "", stem, flags=re.IGNORECASE)
    title = " ".join(w.capitalize() for w in re.split(r"[-_]", slug))
    return slug, title


def _extract_meta(text: str) -> tuple[str, str]:
    os_match = OS_RE.search(text)
    os_val = os_match.group(1).capitalize() if os_match else "Unknown"
    difficulty = "Unknown"
    for line in text.splitlines():
        m = DIFFICULTY_RE.match(line)
        if m:
            difficulty = m.group(1).capitalize()
            break
    return os_val, difficulty


def _clean(text: str) -> str:
    text = IMG_RE.sub("", text)              # drop image embeds
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank runs
    return text.strip()


def _window(text: str, size: int, overlap: int) -> List[str]:
    """Split text longer than `size` into overlapping windows on line breaks."""
    if len(text) <= size:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        # try not to cut mid-line
        if end < len(text):
            nl = window.rfind("\n")
            if nl > size * 0.5:
                window = window[:nl]
                end = start + nl
        out.append(window.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def chunk_file(path: Path) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    slug, title = machine_from_filename(path)
    os_val, difficulty = _extract_meta(raw)

    # Walk the file, grouping lines under their heading breadcrumb.
    sections: List[tuple[list[str], str]] = []  # (heading_stack, body)
    stack: list[tuple[int, str]] = []           # (level, title)
    body: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush():
        if body:
            crumb = [h for _, h in stack]
            sections.append((crumb, "\n".join(body)))

    for line in raw.splitlines():
        stripped = line.lstrip()
        # Track fenced code blocks so `#` comments inside them are never
        # mistaken for Markdown headings.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            body.append(line)
            continue

        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            body.clear()
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            body.append(line)
    flush()

    chunks: List[Chunk] = []
    idx = 0
    for crumb, sec_body in sections:
        cleaned = _clean(sec_body)
        if len(cleaned) < config.MIN_CHUNK_CHARS:
            continue
        heading_path = " / ".join(crumb) if crumb else "(intro)"
        for window in _window(cleaned, config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS):
            if len(window) < config.MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{slug}#{idx}",
                    machine=slug,
                    machine_title=title,
                    os=os_val,
                    difficulty=difficulty,
                    heading_path=heading_path,
                    text=window,
                    source=path.name,
                )
            )
            idx += 1
    return chunks


def iter_raw_files(raw_dir: Path | None = None) -> Iterator[Path]:
    # Only the machine write-ups (htb-<name>.md); skip notes like DATASET.md.
    raw_dir = raw_dir or config.RAW_DIR
    yield from sorted(raw_dir.glob("htb-*.md"))


def build_chunks(raw_dir: Path | None = None) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in iter_raw_files(raw_dir):
        chunks.extend(chunk_file(path))
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
