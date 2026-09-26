#!/usr/bin/env python
"""Evaluate the RAG system against eval/testset.json.

For each question we measure RETRIEVAL quality at the machine level: does the set
of machines surfaced in the top-k retrieved chunks cover the hand-derived answer
key? We report per-question recall and precision plus corpus-wide macro averages.

Optionally (--ask) we also run full synthesis and check that the required terms
(`must_mention`) and at least one expected machine name appear in the answer, and
we flag machine names cited in the answer that are NOT in the retrieved context
(a proxy for hallucinated citations).

Usage:
    python -m eval.evaluate                # retrieval metrics only (fast)
    python -m eval.evaluate --ask          # + synthesis quality (calls the LLM)
    python -m eval.evaluate --ask --k 10
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from app import index as index_mod
from app.retrieve import search
from app.synthesize import synthesize

TESTSET = Path(__file__).parent / "testset.json"


def machines_in_hits(hits) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.machine not in seen:
            seen.append(h.chunk.machine)
    return seen


def prf(expected: set[str], retrieved: set[str]) -> tuple[float, float]:
    if not expected:
        return 0.0, 0.0
    hit = expected & retrieved
    recall = len(hit) / len(expected)
    precision = len(hit) / len(retrieved) if retrieved else 0.0
    return recall, precision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="top-k chunks to retrieve")
    ap.add_argument("--ask", action="store_true", help="also run synthesis checks")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    idx = index_mod.load()
    data = json.loads(TESTSET.read_text())
    questions = data["questions"]

    rows = []
    for q in questions:
        hits = search(idx, q["question"], top_k=args.k)
        retrieved = machines_in_hits(hits)
        # machine-level recall/precision uses the top ceil(k) machines
        exp = {m.lower() for m in q["expected_machines"]}
        got = {m.lower() for m in retrieved}
        recall, precision = prf(exp, got)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "expected": sorted(exp),
            "retrieved_machines": retrieved,
            "hits": sorted(exp & got),
        }

        if args.ask:
            answer = synthesize(q["question"], hits)
            ans_low = answer.lower()
            mentioned = [t for t in q["must_mention"] if t.lower() in ans_low]
            cites_expected = sorted(m for m in exp if re.search(rf"\b{re.escape(m)}\b", ans_low))
            # machine names cited in the answer that are not in retrieved context
            context_machines = {h.chunk.machine.lower() for h in hits}
            all_machines = {c.machine.lower() for c in idx.chunks}
            cited = {m for m in all_machines if re.search(rf"\bseen on:[^)]*\b{re.escape(m)}\b", ans_low)}
            hallucinated = sorted(cited - context_machines)
            row.update({
                "must_mention_hits": f"{len(mentioned)}/{len(q['must_mention'])}",
                "missing_terms": [t for t in q["must_mention"] if t.lower() not in ans_low],
                "cited_expected_machines": cites_expected,
                "possibly_hallucinated_citations": hallucinated,
                "answer": answer,
            })
        rows.append(row)

    macro_recall = mean(r["recall"] for r in rows)
    macro_precision = mean(r["precision"] for r in rows)

    if args.json:
        print(json.dumps({"rows": rows, "macro_recall": macro_recall,
                          "macro_precision": macro_precision}, indent=2))
        return 0

    # human-readable table
    print(f"\nRetrieval @ top-{args.k}  (machine-level vs hand-derived key)\n" + "=" * 72)
    print(f"{'id':<5}{'type':<11}{'recall':>7}{'prec':>7}  hits")
    print("-" * 72)
    for r in rows:
        print(f"{r['id']:<5}{r['type']:<11}{r['recall']:>7.2f}{r['precision']:>7.2f}  "
              f"{','.join(r['hits']) or '-'}")
    print("-" * 72)
    print(f"{'MACRO':<5}{'':<11}{macro_recall:>7.2f}{macro_precision:>7.2f}")

    if args.ask:
        print("\nSynthesis quality\n" + "=" * 72)
        for r in rows:
            flag = "  ⚠ HALLUCINATED: " + ",".join(r["possibly_hallucinated_citations"]) \
                if r["possibly_hallucinated_citations"] else ""
            print(f"{r['id']}: must_mention {r['must_mention_hits']}"
                  f"  cited_expected={','.join(r['cited_expected_machines']) or '-'}{flag}")
            if r["missing_terms"]:
                print(f"      missing terms: {', '.join(r['missing_terms'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
