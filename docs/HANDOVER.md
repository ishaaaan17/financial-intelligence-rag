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
cp .env.example .env   # if an example file exists — otherwise create .env manually
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

> Add anything currently outstanding here.

-

## 7. Next steps

> Add remaining next steps here — the retry-path test and extended `test_pipeline.py` coverage that were previously listed are now complete (see §3).

-

## 8. Key references

- `docs/ARCHITECTURE.md` — system topology, component responsibilities, data flow, latency budget
- `docs/DECISIONS.md` — decision log (e.g. DEC-014 bounded retry, DEC-015 backoff wrapper)
- `docs/CONSTRAINTS.md` — hard limits and operational rules (§11: retry/backoff constraints)
- `docs/FLOW.md` — additional flow documentation
