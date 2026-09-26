# HTB Cheatsheet Assistant (RAG)

A Retrieval-Augmented Generation service that answers natural-language questions about
offensive-security techniques by retrieving relevant passages from the
[HTB write-up corpus](https://github.com/0xh7ml/htb-wiki/tree/main/raw) and having a
local LLM synthesize a **grounded, cited** answer — never a generic answer from the
model's own training data.

> Reference query it handles well:
> **"Provide me the Windows privilege escalation cheatsheet."**
> → groups techniques (SAM/SYSTEM dumps, potato-family, ADCS abuse, ACL/BloodHound…)
> with a short explanation of each and cites the machines that demonstrate them, e.g.
> `(seen on: Absolute, APT)`.

Everything runs **locally through [Ollama](https://ollama.com)** — the corpus never
leaves your machine.

---

## Architecture

```
raw *.md ──▶ ingest (heading-aware chunking) ──▶ index ┌─ BM25 (lexical)
                                                        └─ embeddings (dense, Ollama)
                                                              │
query ──────────────────────────────────────────▶ hybrid retrieval (RRF fusion)
                                                              │  top-k passages
                                                              ▼
                                          synthesize (llama3.2, grounded + cited)
```

* **Retrieval:** hybrid **BM25 + dense embeddings**, fused with Reciprocal Rank Fusion.
* **Embeddings:** `all-minilm` by default (fast, local); `nomic-embed-text` optional.
* **Synthesis:** `llama3.2:latest` (~2 GB) via Ollama, constrained to the retrieved text.

See [`docs/design_note.md`](docs/design_note.md) for the chunking + lexical-vs-embedding
rationale, and [`docs/evaluation.md`](docs/evaluation.md) for the scored results.

---

## Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com) running locally, with the models pulled:

```bash
ollama pull llama3.2:latest      # synthesis LLM (~2 GB)
ollama pull all-minilm           # embeddings (45 MB, default)
# optional, higher-quality/slower embeddings:
# ollama pull nomic-embed-text
```

## Setup

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Fetch the corpus (513 write-ups) into `data/raw/`:

```bash
python -m scripts.fetch_data
```

> The corpus is **not vendored** in this repo: the write-ups embed example CTF
> tokens/keys from retired machines that trip secret scanners. `fetch_data.py`
> pulls the exact same files from the upstream source
> ([0xh7ml/htb-wiki](https://github.com/0xh7ml/htb-wiki/tree/main/raw)), so the
> RAG is fully reproducible. See `data/raw/DATASET.md`.

## Build the index

Either from the CLI:

```bash
python -m scripts.ingest              # hybrid (BM25 + embeddings)
python -m scripts.ingest --no-dense   # lexical only, no embeddings
```

…or over the API once the server is up (`POST /ingest`). Building embeds ~28.6k chunks
and takes a few minutes on CPU with `all-minilm`. The index is written to `data/index/`.

## Run the API

```bash
./scripts/run.sh                      # http://localhost:8000  (Swagger UI at /docs)
# or:  uvicorn app.main:app --reload
```

---

## Endpoints

| Method | Path       | Purpose                                                |
|-------:|------------|--------------------------------------------------------|
| GET    | `/health`  | Liveness, Ollama reachability, index status            |
| GET    | `/stats`   | Corpus/index statistics                                |
| POST   | `/ingest`  | (Re)build the index (`{"force": true}` to rebuild)     |
| GET    | `/search`  | **Retrieval only** — ranked passages, no LLM           |
| POST   | `/ask`     | **Full RAG** — retrieve + grounded, cited answer       |

### Examples

```bash
# Health
curl -s localhost:8000/health | jq

# Build the index (first run)
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"force": true}' | jq

# Retrieval only — see exactly which passages back an answer
curl -s "localhost:8000/search?q=kerberoasting+GetUserSPNs&top_k=5" | jq

# Full RAG answer with citations
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Provide me the Windows privilege escalation cheatsheet."}' | jq
```

`/ask` returns:

```json
{
  "question": "...",
  "answer": "…grouped techniques, each with (seen on: Machine, Machine)…",
  "citations": [{"machine": "Absolute", "os": "Windows", "sections": [...]}],
  "retrieved": 8
}
```

Pass `"include_context": true` to `/ask` (or `include_text=true` to `/search`) to get
the raw retrieved passages back for inspection.

---

## Evaluation

A hand-written test set of 15 questions with a **hand-derived answer key** (built by
grepping the raw files, not by trusting the system) lives in
[`eval/testset.json`](eval/testset.json).

```bash
python -m eval.evaluate            # retrieval recall/precision (fast)
python -m eval.evaluate --ask      # + synthesis quality (calls the LLM)
```

Scored results and analysis: [`docs/evaluation.md`](docs/evaluation.md).

---

## Configuration

All via environment variables (see [`app/config.py`](app/config.py)):

| Variable            | Default              | Meaning                              |
|---------------------|----------------------|--------------------------------------|
| `OLLAMA_HOST`       | `http://localhost:11434` | Ollama endpoint                  |
| `RAG_LLM_MODEL`     | `llama3.2:latest`    | Synthesis model                      |
| `RAG_EMBED_MODEL`   | `all-minilm`         | Embedding model                      |
| `RAG_USE_DENSE`     | `1`                  | `0` = pure lexical (no embeddings)   |
| `RAG_TOP_K`         | `8`                  | Passages fed to the LLM              |

---

## Project layout

```
app/            FastAPI service + RAG pipeline
  config.py       configuration (env-overridable)
  ingest.py       heading-aware Markdown chunking
  index.py        BM25 + dense index build/load/save
  retrieve.py     hybrid retrieval + RRF fusion
  synthesize.py   grounded, cited answer generation
  llm.py          Ollama client (embed + generate)
  main.py         API endpoints
eval/           test set + scoring script
docs/           design note + evaluation writeup
scripts/        fetch_data.py (get corpus), ingest.py (build), run.sh (serve)
data/raw/       the HTB corpus (fetched, not vendored)
```
