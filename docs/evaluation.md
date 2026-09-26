# Evaluation — HTB Cheatsheet Assistant (RAG)

**Setup.** 15 hand-written questions (`eval/testset.json`), answer keys hand-derived by
grepping the raw write-ups (not from the system). Corpus: 513 machines → 28,634 chunks.
Retriever: hybrid BM25 + `all-minilm` dense, RRF fusion. Synthesis: `llama3.2:latest`.
Metrics collected with `python -m eval.evaluate --ask --k 8`.

Retrieval quality is measured at the **machine level**: of the machines that genuinely
demonstrate a technique (the key), how many appear among the machines surfaced in the
top-k retrieved chunks (recall), and how many of the retrieved machines are in the key
(precision).

## Retrieval results (top-k = 8)

| Slice | Recall | Precision |
|-------|:------:|:---------:|
| **All 15 questions** | **0.55** | **0.52** |
| Specific questions (13) | **0.60** | **0.57** |
| Broad "cheatsheet" questions (2) | 0.17 | 0.14 |

Per-question highlights (see `eval/evaluate.py` for the full table):

| id | question theme | recall | prec |
|----|----------------|:---:|:---:|
| q06 | GodPotato | 0.70 | 1.00 |
| q11 | PwnKit / CVE-2021-4034 | 0.80 | 0.80 |
| q12 | DirtyCow | 1.00 | 0.40 |
| q13 | Log4Shell | 0.40 | 1.00 |
| q14 | KrbRelay | 1.00 | 0.33 |
| q02 | Kerberoasting (GetUserSPNs) | 0.67 | 0.80 |
| q07 | ADCS ESC1 (certipy) | 0.22 | 0.33 |
| q01 | **Windows privesc cheatsheet** | 0.00 | 0.00 |

### Reading the numbers

* **Specific, technique-named queries work well** (recall 0.60, precision 0.57). BM25 is
  the workhorse here — it matches the verbatim tokens the write-ups use (`GodPotato`,
  `pkexec`, `GetUserSPNs`), and dense retrieval fills in paraphrases. Precision is high
  on distinctive terms (Log4Shell 1.00, GodPotato 1.00).
* **Broad "cheatsheet" queries are the weak spot** (recall 0.17). The flagship query
  *"Provide me the Windows privilege escalation cheatsheet"* scores **0.00** against its
  key — **but this is largely a measurement artifact, not a retrieval failure.** Roughly
  100+ machines demonstrate some Windows privesc; the hand key is one valid 10-machine
  subset. The retriever returns genuine Windows-privesc sections (Control/WinPEAS,
  Grandpa MS-bulletins, Return/Redelegate token abuse) — correct content that simply
  doesn't intersect the fixed key. The deeper, real issue is that the query contains no
  technique tokens, so BM25 has nothing sharp to grab and `all-minilm` drifts toward
  generic "enumeration" sections rather than the modern potato/ADCS techniques the ideal
  cheatsheet wants. This is the case the "what I'd improve" note targets.
* **Precision < 1.0 is expected and fine.** A machine can appear in the top-8 for a real
  reason yet not be in a *high-precision* key that deliberately lists only unambiguous
  examples (e.g. DirtyCow's key is just 2 machines; the retriever also surfaces other
  kernel-exploit boxes). Precision here undercounts rather than reflecting noise.

## Synthesis quality

Measured on the generated answers:

* **Concept coverage** (`must_mention` terms present in the answer): **72%** overall,
  **80%** on specific questions. The model reliably names the right primitives —
  `GetUserSPNs`/TGS, `SeImpersonate`/named pipe, `certipy`/ESC1, `pkexec`/polkit, etc.
* **Citation correctness:** **14 / 15** answers cite at least one machine that is in the
  hand-derived key, and citations are drawn from the retrieved passages. Example (q06):
  GodPotato correctly attributed to *Job* and *Breach*; (q11) PwnKit to *Antique*,
  *Paper*, *Routerspace*.
* **Grounding / refusal works.** The prompt forbids outside knowledge, and the model
  honours it: for a GodPotato question it explicitly wrote *"Not explicitly mentioned in
  the CONTEXT passages"* for adjacent techniques rather than inventing them.
* **Did it invent anything?** The automated hallucinated-citation check flagged **0 real
  cases** (its single flag, "registry" on q01, is a false positive — the answer wrote
  *"Registry Privilege Escalation … seen on: Control"*, i.e. the Windows registry as a
  technique, not the machine Registry). One genuine minor slip: on q01 the 3B model once
  listed *"Windows 8.1"* — an OS version, not a machine — inside a *seen on:* clause.
  Small models occasionally mislabel a non-machine token as a source; a re-ranker + a
  citation validator against the known machine list would catch this.

## Bottom line

The system does what it was built to do: for concrete offensive-security questions it
retrieves the right passages and produces grounded, correctly-cited answers without
inventing techniques. Its measured weak spot is broad, keyword-free "cheatsheet" queries
— partly a scoring artifact, partly a real retrieval gap that query expansion and
re-ranking (see the design note) would close.
