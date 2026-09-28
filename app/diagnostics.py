"""Deep configuration + connectivity check: "is everything wired up?".

Backs `GET /diagnostics`. Unlike `/health`, every check exercises the real
dependency, so a pass means that part of the pipeline actually works:

    config      effective settings, and values that cannot work
    raw_data    the write-up corpus is present (needed to ingest)
    ollama      the Ollama server is reachable
    models      the configured LLM and embedding models are pulled
    embedding   real embed calls on a short text AND a chunk-sized one (some
                Ollama builds embed short text but reject full-size chunks,
                which fails every ingest batch)
    generation  a tiny real completion from the LLM (the slowest; skippable)
    chromadb    the client connects (embedded store or standalone server)
    index       chunks.jsonl, meta.json and the Chroma collection agree with
                each other and with the configured embedding model
    retrieval   an end-to-end query: embed -> Chroma nearest-neighbour search

Checks are read-only: nothing is built or created (not even an empty embedded
Chroma store). A check that cannot run because an earlier one failed reports
"skip" with the reason.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from urllib.parse import urlparse

from . import config, index as index_mod, llm
from .ingest import iter_raw_files

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

# Stand-in for the largest real chunk: retrieval_text() is a header plus up to
# MAX_CHUNK_CHARS of body. Command-heavy text tokenizes densely, like the corpus.
_LONG_SAMPLE_HEADER = "[Machine: Diagnostics | OS: Linux] Enumeration > Nmap\n"
_LONG_SAMPLE_BODY = "nmap -p- --min-rate 10000 -oA scans/alltcp 10.10.11.42 && "
_PROBE_QUERY = "windows privilege escalation"


@dataclass
class Check:
    status: str
    detail: str
    fix: Optional[str] = None       # what to do about a warn/fail
    data: dict = field(default_factory=dict)
    name: str = ""                  # set by the runner
    ms: float = 0.0                 # set by the runner

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "fix": self.fix,
            "data": self.data,
            "ms": self.ms,
        }


def _model_tag(name: str) -> str:
    """Ollama lists `all-minilm` as `all-minilm:latest`; compare in that form."""
    return name if ":" in name else f"{name}:latest"


def _chroma_mode() -> str:
    return "server" if config.CHROMA_HOST else "embedded"


def _chroma_target() -> str:
    if config.CHROMA_HOST:
        scheme = "https" if config.CHROMA_SSL else "http"
        return f"{scheme}://{config.CHROMA_HOST}:{config.CHROMA_PORT}"
    return str(index_mod._chroma_dir(config.INDEX_DIR))


class _Diagnostics:
    """Runs the checks in order; later checks build on what earlier ones found."""

    def __init__(self, loaded_index, building: bool, run_generation: bool):
        self.loaded_index = loaded_index    # the API's in-memory index, if loaded
        self.building = building
        self.run_generation = run_generation
        # Filled in by earlier checks:
        self.ollama_ok = False
        self.pulled: set[str] = set()
        self.embed_dim: Optional[int] = None
        self.chroma_client = None
        self.collection = None
        self.index_embed_model: Optional[str] = None

    def run(self) -> List[Check]:
        steps: List[tuple[str, Callable[[], Check]]] = [
            ("config", self.check_config),
            ("raw_data", self.check_raw_data),
            ("ollama", self.check_ollama),
            ("models", self.check_models),
            ("embedding", self.check_embedding),
            ("generation", self.check_generation),
            ("chromadb", self.check_chromadb),
            ("index", self.check_index),
            ("retrieval", self.check_retrieval),
        ]
        results: List[Check] = []
        for name, fn in steps:
            start = time.perf_counter()
            try:
                check = fn()
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
                check = Check(FAIL, f"Check crashed: {type(exc).__name__}: {exc}")
            check.name = name
            check.ms = round((time.perf_counter() - start) * 1000, 1)
            results.append(check)
        return results

    def _is_pulled(self, model: str) -> bool:
        return _model_tag(model) in self.pulled

    # --- checks -----------------------------------------------------------
    def check_config(self) -> Check:
        env_file = config.BASE_DIR / ".env"
        data = {
            "env_file": str(env_file) if env_file.exists() else None,
            "ollama_host": config.OLLAMA_HOST,
            "llm_model": config.LLM_MODEL,
            "embed_model": config.EMBED_MODEL,
            "chroma_mode": _chroma_mode(),
            "chroma_target": _chroma_target(),
            "raw_dir": str(config.RAW_DIR),
            "index_dir": str(config.INDEX_DIR),
            "top_k": config.DEFAULT_TOP_K,
            "request_timeout_s": config.REQUEST_TIMEOUT,
        }
        source = ".env + environment" if env_file.exists() else "defaults + environment (no .env file)"

        url = urlparse(config.OLLAMA_HOST)
        if url.scheme not in ("http", "https") or not url.hostname:
            return Check(
                FAIL,
                f"OLLAMA_HOST={config.OLLAMA_HOST!r} is not a URL",
                fix="Set OLLAMA_HOST to a full URL such as http://localhost:11434. "
                "(Ollama's own server also reads OLLAMA_HOST, as a listen address "
                "like 0.0.0.0; this app needs the address to connect to.)",
                data=data,
            )
        if url.hostname == "0.0.0.0":
            return Check(
                WARN,
                "OLLAMA_HOST uses 0.0.0.0, a listen address; connecting to it fails on Windows",
                fix="Use http://localhost:11434 (or the machine's real IP).",
                data=data,
            )
        return Check(PASS, f"Settings loaded from {source}", data=data)

    def check_raw_data(self) -> Check:
        n = sum(1 for _ in iter_raw_files())
        data = {"raw_dir": str(config.RAW_DIR), "markdown_files": n}
        if n:
            return Check(PASS, f"{n} Markdown files in {config.RAW_DIR}", data=data)
        # The corpus is only needed to (re)build; an existing index still works.
        return Check(
            WARN if index_mod.exists() else FAIL,
            f"No Markdown write-ups in {config.RAW_DIR}",
            fix="Fetch the corpus with `python -m scripts.fetch_data`, or point RAG_RAW_DIR at it.",
            data=data,
        )

    def check_ollama(self) -> Check:
        try:
            version = llm.ping().get("version")
        except llm.OllamaError as exc:
            return Check(
                FAIL,
                str(exc),
                fix="Start Ollama (the desktop app, or `ollama serve`) and make sure "
                "OLLAMA_HOST points at it.",
            )
        self.ollama_ok = True
        return Check(PASS, f"Ollama {version} at {config.OLLAMA_HOST}", data={"version": version})

    def check_models(self) -> Check:
        if not self.ollama_ok:
            return Check(SKIP, "Ollama is unreachable")
        self.pulled = {_model_tag(m) for m in llm.list_models()}
        wanted = {"embed_model": config.EMBED_MODEL, "llm_model": config.LLM_MODEL}
        data = {**wanted, "installed": sorted(self.pulled)}
        missing = [m for m in wanted.values() if not self._is_pulled(m)]
        if missing:
            return Check(
                FAIL,
                "Not pulled into Ollama: " + ", ".join(missing),
                fix=" and ".join(f"`ollama pull {m}`" for m in missing),
                data=data,
            )
        return Check(PASS, f"{config.EMBED_MODEL} and {config.LLM_MODEL} are pulled", data=data)

    def check_embedding(self) -> Check:
        model = config.EMBED_MODEL
        if not self.ollama_ok:
            return Check(SKIP, "Ollama is unreachable")
        if not self._is_pulled(model):
            return Check(SKIP, f"{model} is not pulled")

        try:
            short = llm.embed(["nmap -sC -sV 10.10.11.42"], model=model)[0]
        except llm.OllamaError as exc:
            return Check(
                FAIL,
                str(exc),
                fix=f"Ollama cannot run {model} at all. Update Ollama to the latest "
                f"release, re-pull the model (`ollama pull {model}`), and check the "
                "Ollama server log for the underlying error.",
            )

        long_text = (_LONG_SAMPLE_HEADER + _LONG_SAMPLE_BODY * 100)[: config.MAX_CHUNK_CHARS + 200]
        try:
            long = llm.embed([long_text], model=model)[0]
        except llm.OllamaError as exc:
            return Check(
                FAIL,
                f"Short text embeds, but a chunk-sized input ({len(long_text)} chars) fails: {exc}",
                fix="This breaks ingest, since every batch contains full-size chunks. "
                "Update Ollama to the latest release (it truncates over-long inputs "
                "to the model's context window), or switch to an embedding model "
                "with a longer context: set RAG_EMBED_MODEL=nomic-embed-text, run "
                "`ollama pull nomic-embed-text`, then re-ingest.",
            )

        data = {"model": model, "dim": len(short), "long_input_chars": len(long_text)}
        for label, vec in (("short", short), ("chunk-sized", long)):
            if not vec or not any(vec) or not all(math.isfinite(x) for x in vec):
                return Check(FAIL, f"{model} returned an empty, all-zero or non-finite vector "
                             f"for the {label} input", fix="Re-pull the model and update Ollama.", data=data)
        if len(short) != len(long):
            return Check(FAIL, f"{model} returned vectors of different sizes "
                         f"({len(short)} vs {len(long)})", fix="Re-pull the model and update Ollama.", data=data)
        self.embed_dim = len(short)
        return Check(PASS, f"{model} returns {self.embed_dim}-dim vectors for short and "
                     "chunk-sized text", data=data)

    def check_generation(self) -> Check:
        model = config.LLM_MODEL
        if not self.run_generation:
            return Check(SKIP, "Skipped (generate=false)")
        if not self.ollama_ok:
            return Check(SKIP, "Ollama is unreachable")
        if not self._is_pulled(model):
            return Check(SKIP, f"{model} is not pulled")
        try:
            reply = llm.generate("Reply with exactly one word: OK", num_predict=8)
        except llm.OllamaError as exc:
            return Check(
                FAIL,
                str(exc),
                fix=f"Check the Ollama server log. A timeout usually means {model} is "
                "too slow or too large for this machine: raise RAG_REQUEST_TIMEOUT "
                "or choose a smaller RAG_LLM_MODEL.",
            )
        data = {"model": model, "reply": reply[:80]}
        if not reply:
            return Check(WARN, f"{model} returned an empty completion",
                         fix="Re-pull the model and check the Ollama server log.", data=data)
        return Check(PASS, f"{model} generated a reply", data=data)

    def check_chromadb(self) -> Check:
        import chromadb

        target = _chroma_target()
        data = {"mode": _chroma_mode(), "target": target, "client_version": chromadb.__version__}
        if not config.CHROMA_HOST and not index_mod._chroma_dir(config.INDEX_DIR).exists():
            # Opening a PersistentClient would create the store; stay read-only.
            return Check(
                WARN,
                f"No embedded Chroma store at {target} yet (the first ingest creates it)",
                fix="Build the index: `python -m scripts.ingest`.",
                data=data,
            )
        try:
            client = index_mod._chroma_client(config.INDEX_DIR)
            client.heartbeat()
            server_version = client.get_version()
            collections = sorted(getattr(c, "name", c) for c in client.list_collections())
        except Exception as exc:  # noqa: BLE001 - any failure here means "not connected"
            if config.CHROMA_HOST:
                fix = (f"Start the server (`chroma run --host 0.0.0.0 --port {config.CHROMA_PORT} "
                       "--path ./chroma-data`) or fix RAG_CHROMA_HOST / RAG_CHROMA_PORT / "
                       "RAG_CHROMA_SSL in .env.")
            else:
                fix = ("The embedded store may be damaged or written by another chromadb "
                       "version; rebuild it with `python -m scripts.ingest`.")
            return Check(FAIL, f"Cannot open Chroma at {target}: {exc}", fix=fix, data=data)

        self.chroma_client = client
        data.update(server_version=server_version, collections=collections)
        # Only the major version is comparable: 1.x servers report a fixed API
        # version ("1.0.0") whatever chromadb release they ship in.
        if config.CHROMA_HOST and server_version.split(".")[0] != chromadb.__version__.split(".")[0]:
            return Check(
                FAIL,
                f"chromadb client {chromadb.__version__} cannot talk to server "
                f"{server_version} (different major versions)",
                fix=f"Run the server from chromadb=={chromadb.__version__} (the version "
                "in requirements.txt).",
                data=data,
            )
        return Check(PASS, f"Connected to {_chroma_mode()} Chroma {server_version} ({target})", data=data)

    def check_index(self) -> Check:
        if self.building:
            return Check(WARN, "An ingest is running; the index is being rebuilt",
                         fix="Run diagnostics again once the ingest finishes.")
        if not index_mod.exists():
            return Check(FAIL, f"No index at {config.INDEX_DIR}",
                         fix="Build it: `python -m scripts.ingest` (or POST /ingest).")

        if self.loaded_index is not None:
            chunks = self.loaded_index.size
        else:
            with (config.INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as fh:
                chunks = sum(1 for _ in fh)
        meta_file = config.INDEX_DIR / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.index_embed_model = meta.get("embed_model")
        data = {"chunks": chunks, "built_with": self.index_embed_model,
                "built_at": meta.get("built_at")}
        rebuild = "Rebuild the index: `python -m scripts.ingest`."

        if self.chroma_client is None:
            return Check(FAIL, f"chunks.jsonl has {chunks} chunks, but their vectors are "
                         "unavailable (see the chromadb check)", fix=rebuild, data=data)
        try:
            collection = self.chroma_client.get_collection(index_mod.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - chroma raises different types per version
            return Check(
                FAIL,
                f"Chroma collection '{index_mod.COLLECTION_NAME}' does not exist on {_chroma_target()}",
                fix=rebuild + " (If you just switched RAG_CHROMA_HOST between the embedded "
                "store and a Chroma server, the vectors are still on the other one.)",
                data=data,
            )

        vectors = collection.count()
        dim = None
        embs = collection.peek(limit=1).get("embeddings")
        if embs is not None and len(embs):
            dim = len(embs[0])
        space = (collection.metadata or {}).get("hnsw:space")
        data.update(vectors=vectors, dim=dim, space=space)

        if vectors == 0:
            return Check(FAIL, "The Chroma collection is empty", fix=rebuild, data=data)
        if vectors != chunks:
            return Check(FAIL, f"{vectors} vectors in Chroma but {chunks} chunks on disk "
                         "(interrupted or stale ingest)", fix=rebuild, data=data)
        built_with = self.index_embed_model or config.EMBED_MODEL
        if self.ollama_ok and not self._is_pulled(built_with):
            return Check(FAIL, f"The index was built with {built_with}, which is not pulled, "
                         "so queries cannot be embedded",
                         fix=f"`ollama pull {built_with}`, or re-ingest with RAG_EMBED_MODEL.", data=data)
        if self.loaded_index is not None and not self.loaded_index.dense_enabled:
            return Check(FAIL, "The API loaded the index while Chroma was unavailable, so it "
                         "has no vector store in memory", fix="Restart the API.", data=data)
        self.collection = collection

        if _model_tag(built_with) != _model_tag(config.EMBED_MODEL):
            return Check(WARN, f"Index built with {built_with} but RAG_EMBED_MODEL is "
                         f"{config.EMBED_MODEL}; searches keep using {built_with} until you re-ingest",
                         fix="Re-ingest to switch models, or set RAG_EMBED_MODEL back.", data=data)
        if self.embed_dim is not None and dim is not None and dim != self.embed_dim:
            return Check(FAIL, f"Stored vectors are {dim}-dim but {built_with} now returns "
                         f"{self.embed_dim}-dim", fix=rebuild, data=data)
        if space != "cosine":
            return Check(WARN, f"Collection uses '{space}' distance instead of cosine, so scores "
                         "are not cosine similarity", fix=rebuild, data=data)
        return Check(PASS, f"{vectors} vectors match {chunks} chunks ({built_with}, {dim}-dim)", data=data)

    def check_retrieval(self) -> Check:
        if self.collection is None:
            return Check(SKIP, "No usable vector collection (see the index check)")
        if not self.ollama_ok:
            return Check(SKIP, "Ollama is unreachable")
        model = self.index_embed_model or config.EMBED_MODEL
        try:
            q = llm.embed([_PROBE_QUERY], model=model)[0]
        except llm.OllamaError as exc:
            return Check(FAIL, f"Cannot embed the query: {exc}", fix="See the embedding check.")
        res = self.collection.query(
            query_embeddings=[q], n_results=3, include=["distances", "metadatas"]
        )
        top = [
            # Chroma cosine "distance" is 1 - cosine_similarity (as in retrieve.py).
            {"id": cid, "machine": (meta or {}).get("machine_title"), "score": round(1.0 - dist, 4)}
            for cid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
        ]
        data = {"query": _PROBE_QUERY, "embed_model": model, "top": top}
        if not top:
            return Check(FAIL, "The query returned no results", fix="Rebuild the index: "
                         "`python -m scripts.ingest`.", data=data)
        return Check(PASS, f"'{_PROBE_QUERY}' -> {top[0]['machine']} (score {top[0]['score']})", data=data)


def run(loaded_index=None, *, building: bool = False, run_generation: bool = True) -> dict:
    """Run every check and summarise. `ok` is False if any check failed."""
    checks = _Diagnostics(loaded_index, building, run_generation).run()
    counts = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL, SKIP)}
    failed = [c.name for c in checks if c.status == FAIL]
    warned = [c.name for c in checks if c.status == WARN]
    if failed:
        summary = "Failing: " + ", ".join(failed)
    elif warned:
        summary = "Working, with warnings: " + ", ".join(warned)
    else:
        summary = "Everything is configured and connected"
    return {
        "ok": not failed,
        "summary": summary,
        "counts": counts,
        "checks": [c.to_dict() for c in checks],
    }
