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
raw *.md ──▶ ingest (heading-aware chunking) ──▶ embeddings ──▶ ChromaDB (dense vectors)
                                                                      │
query ──▶ embed ─────────────────────────────────────▶ nearest-neighbour search
                                                                      │  top-k passages
                                                                      ▼
                                          synthesize (llama3.2, grounded + cited)
```

* **Retrieval:** **embedding (dense) only** — cosine nearest-neighbour search, no lexical component.
* **Vector store:** **ChromaDB** (persistent, cosine space) holds the vectors.
* **Embeddings:** `all-minilm` by default (fast, local); `nomic-embed-text` optional.
* **Synthesis:** `llama3.2:latest` (~2 GB) via Ollama, constrained to the retrieved text.

See [`docs/design_note.md`](docs/design_note.md) for the chunking + embedding-retrieval
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
python -m scripts.ingest              # embeds all chunks into ChromaDB
```

…or over the API once the server is up (`POST /ingest`). Building embeds ~28.6k chunks
and takes a few minutes on CPU with `all-minilm`. The corpus + metadata is written to
`data/index/chunks.jsonl` and the vectors to a ChromaDB store at `data/index/chroma/`.

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
| GET    | `/diagnostics` | **Deep check** that every setting works and is connected |
| GET    | `/stats`   | Corpus/index statistics                                |
| POST   | `/ingest`  | (Re)build the index (`{"force": true}` to rebuild)     |
| GET    | `/search`  | **Retrieval only** — ranked passages, no LLM           |
| POST   | `/ask`     | **Full RAG** — retrieve + grounded, cited answer       |

### Examples

```bash
# Health
curl -s localhost:8000/health | jq

# Is everything configured and connected? (HTTP 503 if any check fails)
curl -s localhost:8000/diagnostics | jq

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

### Troubleshooting with `/diagnostics`

If ingest or a query fails, start the API and open
<http://localhost:8000/diagnostics>. It runs each dependency for real and reports
`pass` / `warn` / `fail` / `skip` per check, with a `fix` hint for anything that
is wrong:

| Check        | What it verifies                                                        |
|--------------|-------------------------------------------------------------------------|
| `config`     | Effective settings (and whether `.env` was found); `OLLAMA_HOST` is a URL |
| `raw_data`   | The Markdown corpus is in `RAG_RAW_DIR`                                 |
| `ollama`     | The Ollama server answers, and its version                              |
| `models`     | `RAG_EMBED_MODEL` and `RAG_LLM_MODEL` are pulled                        |
| `embedding`  | Real embeds of a short **and** a chunk-sized text (catches Ollama builds that reject long inputs, which fails every ingest batch) |
| `generation` | A tiny real LLM completion (skip it with `?generate=false`; it is the slowest) |
| `chromadb`   | Chroma opens/connects (embedded store or `RAG_CHROMA_HOST` server)      |
| `index`      | Vectors in Chroma match `chunks.jsonl`, same embedding model and size   |
| `retrieval`  | An end-to-end search: embed a query, nearest neighbours from Chroma     |

The response has `ok` (false if any check failed), a one-line `summary`, and the
per-check details. The checks only read: they never build, rebuild or create
anything.

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

Settings live in a `.env` config file at the project root. Copy the template and edit it:

```bash
cp .env.example .env      # then edit .env
```

The app loads `.env` automatically on startup (real environment variables, if set,
still take precedence). Available settings:

| Variable            | Default              | Meaning                              |
|---------------------|----------------------|--------------------------------------|
| `OLLAMA_HOST`       | `http://localhost:11434` | Ollama endpoint                  |
| `RAG_LLM_MODEL`     | `llama3.2:latest`    | Synthesis model                      |
| `RAG_EMBED_MODEL`   | `all-minilm`         | Embedding model (`nomic-embed-text` for higher quality) |
| `RAG_TOP_K`         | `8`                  | Passages retrieved and fed to the LLM |
| `RAG_EMBED_BATCH_SIZE` | `384`             | Chunks per embed request (lower on a small machine) |
| `RAG_EMBED_WORKERS` | `3`                  | Concurrent embed requests during ingest |
| `RAG_CHROMA_HOST`   | *(unset)*            | Standalone Chroma server host; unset = embedded on-disk |
| `RAG_CHROMA_PORT`   | `8000`               | Standalone Chroma server port           |
| `RAG_CHROMA_SSL`    | `0`                  | `1` to connect to the Chroma server over HTTPS |
| `RAG_CHROMA_HNSW_M` | `32`                 | HNSW graph degree (index quality)       |
| `RAG_CHROMA_CONSTRUCTION_EF` | `200`       | HNSW build breadth (index quality)      |
| `RAG_CHROMA_SEARCH_EF` | `128`             | HNSW query breadth (search recall)      |

> **Why the HNSW settings exist:** ChromaDB's default index parameters gave poor,
> non-deterministic nearest-neighbour recall on this corpus (some rebuilds returned
> near-random passages). These tuned defaults make retrieval reliable; ingest also
> runs a self-check that fails loudly rather than serve a bad index.

### Using a standalone ChromaDB server (optional)

By default Chroma runs **embedded** (in-process) and persists to `data/index/chroma/`.
To use a separate ChromaDB server instead, start one and set its host/port in `.env`:

```bash
# terminal 1 — run the Chroma server (the `chroma` CLI ships with the requirements)
chroma run --host 0.0.0.0 --port 8001 --path ./chroma-data
```

```bash
# .env — point the app at that server
RAG_CHROMA_HOST=localhost
RAG_CHROMA_PORT=8001
```

```bash
# terminal 2 — ingest + serve; vectors now live on the server, not local disk
python -m scripts.ingest
./scripts/run.sh
```

With `RAG_CHROMA_HOST` set the app connects over HTTP (`HttpClient`), so the vectors
live on the server and are shared by any client that points at it. Note the Chroma
server's own default port is `8000`, which collides with the API's default port — run
one of them elsewhere (the example uses `8001` for Chroma).

---

## Project layout

```
app/            FastAPI service + RAG pipeline
  config.py       configuration (env-overridable)
  ingest.py       heading-aware Markdown chunking
  index.py        ChromaDB vector store: build/load/save
  retrieve.py     embedding (dense) retrieval via ChromaDB
  synthesize.py   grounded, cited answer generation
  llm.py          Ollama client (embed + generate)
  diagnostics.py  deep config/connectivity checks behind /diagnostics
  main.py         API endpoints
eval/           test set + scoring script
docs/           design note + evaluation writeup
scripts/        fetch_data.py (get corpus), ingest.py (build), run.sh (serve)
data/raw/       the HTB corpus (fetched, not vendored)
```
