"""Grounded answer synthesis.

The LLM is instructed to answer ONLY from the retrieved passages and to cite the
machines each technique was seen on, matching the reference format
"(seen on: Absolute, APT)". If the context does not support an answer, it must
say so rather than fall back on its own training data.
"""
from __future__ import annotations

from typing import List

from . import config, llm
from .retrieve import Hit

SYSTEM_PROMPT = (
    "You are the HTB Cheatsheet Assistant. You answer offensive-security "
    "questions ONLY using the provided CONTEXT passages, which are excerpts from "
    "Hack The Box machine write-ups. Rules:\n"
    "1. Use only facts present in the CONTEXT. Never use outside knowledge.\n"
    "2. If the CONTEXT does not contain the answer, say: 'The provided write-ups "
    "do not cover this.' and stop.\n"
    "3. For cheatsheet-style questions, group techniques with a short explanation "
    "of each.\n"
    "4. After each technique, cite the machine(s) it was seen on, e.g. "
    "'(seen on: Absolute, APT)'. Use only machine names that appear in the "
    "CONTEXT for that technique.\n"
    "5. Be concise and technical. Do not invent commands, tools, or machine names."
)


def build_context(hits: List[Hit], max_chars: int = 6000) -> str:
    blocks: List[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        block = (
            f"[Passage {i}] Machine: {h.chunk.machine_title} | OS: {h.chunk.os} | "
            f"Section: {h.chunk.heading_path}\n{h.chunk.text}"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above. Group techniques, explain each "
        "briefly, and cite the machine(s) for each technique in the form "
        "'(seen on: Name, Name)'. If the CONTEXT is insufficient, say so."
    )


def synthesize(question: str, hits: List[Hit]) -> str:
    if not hits:
        return "The provided write-ups do not cover this."
    context = build_context(hits)
    prompt = build_prompt(question, context)
    return llm.generate(prompt, system=SYSTEM_PROMPT)


def citations(hits: List[Hit]) -> List[dict]:
    """Deduplicated machine list backing the answer, for the API response."""
    seen: dict[str, dict] = {}
    for h in hits:
        if h.chunk.machine not in seen:
            seen[h.chunk.machine] = {
                "machine": h.chunk.machine_title,
                "os": h.chunk.os,
                "source": h.chunk.source,
                "sections": [],
            }
        seen[h.chunk.machine]["sections"].append(h.chunk.heading_path)
    return list(seen.values())
