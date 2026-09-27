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
the embedder an anchor the raw body often lacks (a code block rarely repeats the
machine name), and it is what lets the system answer "which machine(s)?".

Result: ~28.6k chunks over 513 machines, ~1.0k chars each.

## Retrieval — embedding (dense) only

Retrieval is a **pure vector search**. The query is embedded with the same model as
the corpus, and **ChromaDB** returns the nearest passages by cosine similarity. There
is no lexical/keyword component.

Why embeddings rather than lexical:

* **Meaning over spelling.** Embeddings match intent even when the words differ, so
  broad questions ("how do I become SYSTEM from a service account", "windows privilege
  escalation cheatsheet") retrieve relevant passages that share no literal tokens with
  the query — where a keyword search returns nothing.
* **One representation, one store.** ChromaDB owns the vectors, metadata, and the
  approximate-nearest-neighbour index; it persists to disk and survives restarts
  without re-embedding. The retrieval path is a single `collection.query()`.

The honest trade-off: dropping lexical search costs precision on queries built around a
verbatim token (`GetUserSPNs`, `pkexec`, `ESC1`), where an exact-string matcher is hard
to beat and the embedder can blur near-neighbours (`ESC1` vs `ESC8`). Recovering that is
the top item under "what I'd improve."

Embeddings run locally through Ollama. I default to **all-minilm** (45 MB, 384-dim):
on this CPU it embeds ~10× faster than `nomic-embed-text` (~19 vs ~1.8 chunks/s) and is
accurate enough for this corpus. `nomic-embed-text` is a one-env-var swap
(`RAG_EMBED_MODEL`) when higher-quality embeddings are worth the slower build.

## Synthesis / grounding

The LLM (`llama3.2:latest`, ~2 GB, via Ollama) is given only the retrieved passages
and a system prompt that forbids outside knowledge, requires grouping techniques with
short explanations, and requires a `(seen on: Name, Name)` citation per technique
drawn only from the passages. If the context is insufficient it must say so. The API
also returns a structured `citations` list of the machines actually retrieved, so a
caller can verify the answer against its sources.

## What I'd improve with another week

**Re-add lexical signal, then re-rank.** Pure embedding retrieval loses precision on
verbatim-token queries. I'd bring back a keyword retriever (BM25) *and* fuse it with the
dense results (RRF) to recover exact-term precision, then run a lightweight re-ranker
(even `llama3.2` scoring query–passage relevance, or a small cross-encoder) over the top
~30 candidates before the LLM sees the top ~8. The re-rank would also let me group by
*technique* across machines rather than by raw similarity — exactly the shape a good
cheatsheet answer wants.
