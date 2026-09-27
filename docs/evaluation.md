# Evaluation — HTB Cheatsheet Assistant (RAG)

**Setup.** 15 hand-written questions (`eval/testset.json`), answer keys hand-derived by
grepping the raw write-ups (not from the system). Corpus: 513 machines → 28,634 chunks.
Retriever: **embedding (dense) only** — ChromaDB cosine nearest-neighbour over
`all-minilm` vectors. Synthesis: `llama3.2:latest`. Metrics collected with
`python -m eval.evaluate --ask --k 8`.

Retrieval quality is measured at the **machine level**: of the machines that genuinely
demonstrate a technique (the key), how many appear among the machines surfaced in the
top-k retrieved chunks (recall), and how many of the retrieved machines are in the key
(precision).

## Retrieval results (top-k = 8)

| Slice | Recall | Precision |
|-------|:------:|:---------:|
| **All 15 questions** | **0.43** | **0.39** |
| Specific questions (13) | **0.48** | **0.43** |
| Broad "cheatsheet" questions (2) | 0.13 | 0.13 |

Per-question highlights (full table from `eval/evaluate.py`):

| id | question theme | recall | prec |
|----|----------------|:---:|:---:|
| q12 | DirtyCow | 1.00 | 0.40 |
| q14 | KrbRelay | 1.00 | 0.33 |
| q06 | GodPotato | 0.60 | 0.75 |
| q11 | PwnKit / CVE-2021-4034 | 0.60 | 0.50 |
| q15 | SAM/SYSTEM dump | 0.50 | 0.60 |
| q02 | Kerberoasting (GetUserSPNs) | 0.33 | 0.33 |
| q05 | PrintSpoofer | 0.00 | 0.00 |
| q01 | Windows privesc cheatsheet | 0.10 | 0.12 |

### Reading the numbers

* **Embedding-only retrieval trades precision for simplicity.** With one representation
  and one store (ChromaDB), the pipeline is a single vector query — but the results are
  ranked purely by semantic similarity. On this test set that lands at recall 0.48 /
  precision 0.43 on specific questions.
* **The clearest cost is verbatim-token queries.** `q05` (PrintSpoofer) drops to **0.00**:
  the three write-ups that name it are found by an exact-string match far more reliably
  than by meaning, and semantic search pulls in adjacent "SeImpersonate/potato" passages
  from other machines instead. `q02` (GetUserSPNs) slips for the same reason. This is the
  expected downside of removing lexical search, and it is the top item in the design
  note's "what I'd improve."
* **Semantic strengths still show.** `q12` (DirtyCow) and `q14` (KrbRelay) hit 1.00 recall,
  and `q06` (GodPotato) reaches 0.75 precision — the model groups genuinely related
  passages even when phrasing varies.
* **Broad "cheatsheet" queries remain the weak spot** (recall 0.13). A query with no
  technique tokens ("Windows privilege escalation cheatsheet") gives the embedder only a
  vague direction, so it drifts toward generic "enumeration" passages. Partly a scoring
  artifact too: ~100+ machines demonstrate some privesc, and the hand key lists one valid
  subset.

## Synthesis quality

Measured on the generated answers:

* **Concept coverage** (`must_mention` terms present in the answer): **70%** overall,
  **80%** on specific questions — the model still names the right primitives
  (`GetUserSPNs`/TGS, `certipy`/ESC1, `pkexec`/polkit, `log4j`/JNDI).
* **Citation correctness:** **13 / 15** answers cite at least one machine from the
  hand-derived key, drawn from the retrieved passages (e.g. q06 GodPotato → *Breach,
  Haze, Job, Media*; q11 PwnKit → *Antique, Paper*).
* **Grounding / refusal works.** The prompt forbids outside knowledge and the model
  honours it, writing "not covered" for techniques absent from the context rather than
  inventing them.
* **Did it invent citations?** The automated check flagged 2 questions, but both are
  **false positives**: the flagged "machines" are the words *code* and *response* — real
  HTB machine names that also appear as ordinary words in the answer prose. No confirmed
  invented machine citation. (A stricter validator that only accepts names inside the
  `(seen on: …)` parentheses would remove this noise.)

## Comparison: this is a deliberate downgrade from hybrid

An earlier version fused embeddings with a BM25 keyword retriever (RRF). For reference,
that hybrid scored higher on specific questions (recall ~0.60 / precision ~0.57 at k=8).
Switching to **embedding-only** was an intentional design change; the ~0.12 precision it
costs on specific queries is the price of dropping the lexical half, and re-adding it
(then re-ranking) is the first item under "what I'd improve" in the design note.

## Bottom line

Pure embedding retrieval keeps the system grounded and correctly cited, and it shines on
questions where meaning matters more than exact wording. Its measured weakness is queries
built around a specific verbatim token (PrintSpoofer, GetUserSPNs) and broad keyword-free
"cheatsheet" queries — exactly where a lexical retriever would have helped.
