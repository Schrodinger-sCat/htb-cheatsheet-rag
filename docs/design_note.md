# Design Note — HTB Cheatsheet Assistant (RAG)

## Chunking strategy

Each of the 513 write-ups is one HTB machine (`data/raw/htb-<machine>.md`). I chunk
**heading-aware**: the Markdown is parsed into sections following its own `#`…`######`
heading hierarchy, and each chunk carries the full heading **breadcrumb**
(e.g. `Auth as d.klay / AS-Rep-Roast / Capture Hash`) plus the machine name and OS.

Two details that mattered in practice:

1. **Fence tracking.** These write-ups are dense with terminal output, and comment
   lines inside code fences (`# extended LDIF`, `# nmap …`) were being parsed as
   Markdown headings, shattering the breadcrumb. The parser now tracks ```` ``` ````
   / `~~~` fences and ignores `#` lines inside them.
2. **Windowing.** Sections longer than ~1400 chars (big tool dumps) are split into
   overlapping windows (200-char overlap) so no single chunk dominates the embedder
   or the LLM context.

Every chunk is indexed and embedded as `retrieval_text()` =
`[Machine: X | OS: Y] <breadcrumb>\n<body>`. Prepending the machine + section gives
both retrievers an anchor the raw body often lacks (a code block rarely repeats the
machine name), and it is what lets the system answer "which machine(s)?".

Result: ~28.6k chunks over 513 machines, ~1.0k chars each.

## Lexical vs. embedding retrieval — why **hybrid**

Offensive-security queries are bimodal:

* **Exact tokens** the authors write verbatim — `GetUserSPNs`, `SeImpersonatePrivilege`,
  `certipy`, `ESC1`, `pkexec`. Here **BM25** is unbeatable: an embedding model can
  blur `ESC1` vs `ESC8`, but BM25 matches the literal string.
* **Paraphrasable intent** — "windows privilege escalation cheatsheet",
  "how do I become SYSTEM from a service account". Here **dense embeddings** win
  because the write-ups rarely contain the query's exact words.

So I run both and fuse with **Reciprocal Rank Fusion** (RRF, k=60). RRF combines the
two rank lists without having to calibrate BM25 scores against cosine similarities —
robust and parameter-light. BM25 is the backbone (it alone answers most specific
technique questions); embeddings add recall on the broad "cheatsheet" queries.

The dense vectors live in **ChromaDB** (a persistent collection in cosine space at
`data/index/chroma/`). Chroma owns vector storage, metadata, and the nearest-neighbour
search, so retrieval scales past a brute-force NumPy scan and the index survives restarts
without re-embedding. BM25 stays in-memory (it rebuilds from `chunks.jsonl` in a second),
and RRF fuses the two — so ChromaDB is exactly the dense/embedding half of the hybrid.

Embeddings run locally through Ollama. I default to **all-minilm** (45 MB, 384-dim):
on this CPU it embeds ~10× faster than `nomic-embed-text` (~19 vs ~1.8 chunks/s) while
the BM25 half carries exact-term precision. `nomic-embed-text` remains a one-env-var
swap (`RAG_EMBED_MODEL`) when higher-quality embeddings are worth the slower build,
and dense retrieval can be turned off entirely (`RAG_USE_DENSE=0`) for a pure-lexical,
zero-embedding-cost mode.

## Synthesis / grounding

The LLM (`llama3.2:latest`, ~2 GB, via Ollama) is given only the retrieved passages
and a system prompt that forbids outside knowledge, requires grouping techniques with
short explanations, and requires a `(seen on: Name, Name)` citation per technique
drawn only from the passages. If the context is insufficient it must say so. The API
also returns a structured `citations` list of the machines actually retrieved, so a
caller can verify the answer against its sources.

## What I'd improve with another week

**Add a cross-encoder / LLM re-ranker over the fused top-30.** RRF gets the right
passages into the candidate pool, but for broad cheatsheet queries the *ordering*
within the pool is coarse, and the LLM only sees the top ~8. A lightweight re-rank
(even `llama3.2` scoring query–passage relevance, or a small cross-encoder) before
truncation would raise precision on the passages that actually reach synthesis, and
would let me group by *technique* across machines rather than by raw chunk score —
which is exactly the shape a good cheatsheet answer wants.
