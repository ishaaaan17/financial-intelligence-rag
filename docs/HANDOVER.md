# Handover

Project status and orientation notes for the SEC Financial Intelligence RAG Engine. See `ARCHITECTURE.md` for full system design, `DECISIONS.md` for the decision log (DEC-XXX references), and `CONSTRAINTS.md` for hard limits and operational rules.

## 1. Project summary

A RAG system for querying SEC financial filings (10-Q/10-K). A Streamlit frontend talks to a FastAPI backend, which runs a LangGraph state machine that decomposes a query, retrieves metadata-filtered chunks from Qdrant, reranks locally with FlashRank, synthesizes a comparison via Groq, and verifies every numerical claim against retrieved context before responding — with a bounded retry loop if grounding verification fails.

## 2. Current status

- Core query pipeline (decompose → retrieve → rerank → grade → synthesize → verify → respond) is implemented end-to-end in `src/graph.py` / `src/nodes.py`.
- Bounded hallucination-retry loop (`MAX_GENERATION_RETRIES`, DEC-014) and logged retry/backoff wrapper for transient Groq errors (DEC-015) are implemented, per `CONSTRAINTS.md` §11.
- Ingestion pipeline (`src/parser.py`) handles PDF text + table extraction via `pdfplumber` and chunking.
- Vector store isolation by `ticker` / `fiscal_year` / `fiscal_period` is in place in `src/database.py` to prevent cross-quarter context contamination.

## 3. Recently completed

- **Retry-path test** — test coverage for the `increment_retry` → `generate_comparison` loop, confirming the graph correctly re-attempts synthesis on a failed grounding check and falls through to `handle_hallucination` once `MAX_GENERATION_RETRIES` is exhausted.
- **Extended `test_pipeline.py` coverage** — broader coverage of the end-to-end query pipeline beyond the happy path, including the retry scenario above.
- `docs/ARCHITECTURE.md` updated to document the retry loop in the data-flow section and reflect retry cost in the latency budget table.

## 4. Environment setup

> Fill in if these differ from the defaults below.

```powershell
# Clone and enter the repo
git clone https://github.com/ishaaaan17/financial-intelligence-rag.git
cd financial-intelligence-rag

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (see below)
cp .env.example .env   # .env.example is committed in the repo root
```

### Required environment variables

> Names only — set actual values locally, never commit them.

| Variable                                   | Purpose                               |
| ------------------------------------------ | ------------------------------------- |
| `GROQ_API_KEY`                             | Inference (`llama-3.3-70b-versatile`) |
| `QDRANT_URL`                               | Qdrant Cloud endpoint                 |
| `QDRANT_API_KEY`                           | Qdrant Cloud auth                     |
| _(add any others used in `src/config.py`)_ |                                       |

### Running locally

```powershell
# Backend
uvicorn src.app:app --reload

# Frontend
streamlit run frontend.py
```

## 5. Running tests

```powershell
pytest tests/
```

Key test files:

- `tests/test_nodes.py` — unit tests for individual graph nodes
- `tests/test_pipeline.py` — end-to-end pipeline tests, including the retry path

## 6. Known issues / open items

- **No recorded benchmark or eval yet.** `benchmarks/benchmark_retrieval.py`
  exists and measures retrieval-only latency against a real, populated
  Qdrant, but no run has been recorded and no question/answer eval set
  exists. Until they do, no latency or retrieval-accuracy percentage
  should be quoted for this project (see `benchmarks/README.md`).
- **Hosting migration is documented but not yet deployed.** The old EC2
  instance is down; the Vercel frontend + HF Spaces/Render API plan in
  the README's Hosting section still needs the actual deployments and
  their URLs added.
- **Module-level singletons in `src/nodes.py`.** `llm`, `vector_store`
  and `ranker` are constructed at import time (embedding model load,
  Qdrant connection, FlashRank download to a hardcoded `/tmp/flashrank`,
  which also breaks on native Windows). Importing the app therefore
  requires a valid `GROQ_API_KEY` and network access; a lazy-initialised
  factory would fix testability and the Windows path.
- **Docker cold start.** The image pre-downloads the embedding model but
  not the FlashRank reranker, so the first query in a fresh/offline
  container downloads it at runtime. `docker-compose.yml` also has no
  healthchecks and no local Qdrant service (Qdrant Cloud only).
- **No auth or upload size limit on `/ingest` / `/ingest-edgar`.** Fine
  for a local demo; must be addressed before any public deployment.
- **Grounding verification is self-checking.** `verify_grounding` uses
  the same LLM to judge its own output via prompt; there is no
  deterministic numeric cross-check of figures against source tables.
- **Grading is sampled.** `grade_documents` grades only the first 3
  retrieved docs, truncated to 400 chars each — a deliberate cost/latency
  trade-off, documented here so it isn't mistaken for full grading.

## 7. Next steps

- Deploy the API (HF Spaces or Render, using the existing `Dockerfile`),
  deploy the Vercel frontend, and add the live URLs to the README.
- Run `benchmarks/benchmark_retrieval.py` against the populated Qdrant
  and commit the resulting `benchmarks/results/retrieval_latency.json`.
- Build a small eval: 10–20 questions over 2–3 real filings with expected
  tables, reporting retrieval hit-rate at the pipeline's actual top-3.
- Lazy-initialise the `nodes.py` singletons and make the FlashRank cache
  directory configurable via environment variable.
- Add an upload size limit (and an API key check if the API goes public).

## 8. Key references

- `docs/ARCHITECTURE.md` — system topology, component responsibilities, data flow, latency budget
- `docs/DECISIONS.md` — decision log (e.g. DEC-014 bounded retry, DEC-015 backoff wrapper)
- `docs/CONSTRAINTS.md` — hard limits and operational rules (§11: retry/backoff constraints)
- `docs/FLOW.md` — additional flow documentation
