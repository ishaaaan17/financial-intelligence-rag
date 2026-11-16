# 📈 SEC Multi-Hop Financial Intelligence RAG Engine

[![CI Pipeline](https://github.com/ishaaaan17/financial-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ishaaaan17/financial-intelligence-rag/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](<https://img.shields.io/badge/Groq%20(Llama--3.3--70B)-F05A28.svg>)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## System Overview

This is a Retrieval-Augmented Generation system for one specific, awkward kind of question: **comparative questions across SEC filing periods** — *"Compare Apple's R&D spend between Q2 2024 and Q2 2025."* A generic RAG pipeline handles these badly in two characteristic ways:

1. **Cross-period contamination.** Similarity search alone happily retrieves the right metric from the *wrong quarter*. The answer looks fluent and is quietly wrong.
2. **Ungrounded numbers.** An LLM asked to compare figures will produce a table whether or not the retrieved text actually contains those figures.

So the pipeline is built around a single rule — *every answer is a claim that has to survive its own citations* — implemented as a LangGraph state machine:

1. **Decompose** the question into deterministic metadata targets (`ticker`, `years`, `quarters`, `metric`) instead of embedding the raw question and hoping.
2. **Retrieve per period, with hard payload filters** on `ticker` + `fiscal_year` + `fiscal_period`, so one period's chunks can never leak into another period's context.
3. **Rerank** each period's candidates with a local cross-encoder (FlashRank), preferring financial tables over narrative boilerplate.
4. **Grade** the retrieved set; if a period's data is missing, say so instead of improvising.
5. **Generate** the comparison, then **verify** it: a grounding node re-checks the answer's numerical claims against the retrieved filing text, with one regeneration retry before the answer is flagged rather than shipped silently.

The stack is deliberately zero-dollar: local HuggingFace embeddings (`all-MiniLM-L6-v2`), Qdrant Cloud free tier, and Groq's free API tier (`llama-3.3-70b-versatile`, temperature 0.0).

## Measurement Status — What Is and Is Not Claimed

This project previously quoted retrieval numbers (a "7.25 ms P50 latency", "100% Top-4 table retrieval") that were never measured anywhere in the repository. They have been removed, and this section exists so they are not replaced by new ones:

| Claim type | Status |
|---|---|
| End-to-end latency | A **budget**, not a measurement: ~2.5–4.5 s per 2-period comparison, broken down in `docs/ARCHITECTURE.md` §4. Synthesis on Groq (~1–2.5 s) is the dominant cost. |
| Retrieval latency | **Measurable, not yet measured.** `benchmarks/benchmark_retrieval.py` times filtered search against a real, populated Qdrant collection. Until it is run against one, no latency number is quoted. See `benchmarks/README.md` for exactly what it does and does not measure. |
| Metadata isolation | **Enforced and test-covered**, not benchmarked as a percentage: `search_filtered` always builds the ticker/year/period payload filter, and `tests/test_metadata_isolation.py` fails if it ever doesn't. |
| Answer / retrieval accuracy | **No labelled evaluation set exists**, so no accuracy percentage is claimed anywhere. The grounding verifier (below) is a per-answer check, not an accuracy benchmark. |
| Test suite | 37 tests across 5 files (`pytest`), covering chunking, EDGAR ingestion (mocked HTTP), metadata isolation, graph nodes, and the end-to-end pipeline (mocked graph). |

---

## 🚀 Core Architectural Features

### Deterministic Query Decomposition (`src/nodes.py: decompose_query`)

Comparative questions are parsed into structured targets — ticker, the list of (year, quarter) periods, and the metric — before any vector search happens. Retrieval then runs **once per period** against that period's metadata, which turns "compare Q2 2024 with Q2 2025" into two small, well-scoped searches instead of one fuzzy one. State is passed node to node through a typed `AgentState`, so every step's inputs and outputs are inspectable.

### Metadata-Isolated Vector Search (`src/database.py`)

Every search goes through `search_filtered`, which **always** attaches a Qdrant payload filter on `ticker`, `fiscal_year`, and `fiscal_period` — there is no code path that searches the collection unfiltered. The collection auto-creates payload indexes on those fields. This is the mechanism that prevents cross-quarter contamination, and it is locked in by tests: `tests/test_metadata_isolation.py` asserts the filter is built for every call.

### Table-Preserving Ingestion + Real Prose Chunking (`src/parser.py`)

SEC filings live in their tables, so ingestion treats tables and prose differently:

- **Tables** are extracted with `pdfplumber` and serialized to Markdown **as single, unsplittable chunks** — a table cut in half is a wrong number waiting to happen.
- **Prose** is split into overlapping chunks of **~1,200 characters with ~150 characters of overlap** (`CHUNK_SIZE`/`CHUNK_OVERLAP`), so a fact straddling a boundary survives whole in at least one chunk.

Every chunk carries its `ticker` / `fiscal_year` / `fiscal_period` / `page` / `type` metadata into Qdrant — the metadata the isolation filter and the citations are built from.

### FlashRank Reranking (`src/nodes.py: retrieve_filtered_docs`)

Per period, filtered search retrieves 5 candidates; a local FlashRank cross-encoder (`ms-marco-TinyBERT-L-2-v2`) reranks them down to the **top 3** that actually reach the generator. Reranking runs locally (no API cost, no extra network hop) and is what lets dense financial tables outrank the boilerplate disclosure narrative that surrounds them.

### Grading and the Missing-Data Path (`grade_documents`, `handle_missing_data`)

After retrieval, a grading node checks whether the retrieved context is actually relevant. If a requested period has no usable data, the graph routes to `handle_missing_data` and the response says that period is missing — a boring, correct answer instead of a confident, invented one.

### Grounding Verification Loop (`verify_grounding`)

The generated comparison is not returned directly. `verify_grounding` cross-references the answer's **numerical claims** against the raw retrieved context. If a claim is ungrounded and the retry budget remains (`retry_count < MAX_GENERATION_RETRIES`), the graph loops through `increment_retry` back to generation once more; if it is still ungrounded after that, the response is routed to `handle_hallucination` and returned with `hallucination_detected: true` rather than presented as fact. It is an LLM-based check, not a formal proof — see *Known Limitations*.

### SEC EDGAR Ingestion (`src/edgar.py`)

Filings no longer have to be downloaded and uploaded by hand. The EDGAR client resolves ticker → CIK → latest filing of a requested form and saves the primary document into `data/raw/`:

```bash
python scripts/fetch_filing.py --ticker AAPL --form 10-Q
# or via the API: POST /ingest-edgar  { "ticker": "AAPL", "form": "10-Q" }
```

Manual PDF upload (`POST /ingest`) still works, and it fails honestly: if Qdrant rejects the write, the endpoint returns the real **500** error — an earlier version swallowed storage failures and reported success for filings that were never stored.

---

## 🛠️ System Architecture & Workflow

```
           +-----------------------------+
           | User Prompt / API Request   |
           +--------------+--------------+
                          |
                          v
            +----------------------------+
            |   decompose_query Node     |
            | (Extract Ticker & Quarters)|
            +-------------+--------------+
                          |
                          v
            +----------------------------+
            |   Filtered Vector Search   |
            |     (Qdrant Cloud)         |
            +-------------+--------------+
                          |
                          v
            +----------------------------+
            |    FlashRank Reranking     |
            +-------------+--------------+
                          |
                          v
       +----+    grade_documents Node    +----+
       |    +----------------------------+    |
[Irrelevant / Missing]                 [Relevant Context]
       |                                      |
       v                                      v
+-----------------------+              +----------------------+
|  handle_missing_data  |              | generate_comparison  |
+----------+------------+              +----------+-----------+
           |                                      |
           |                                      v
           |                           +----------------------+
           |                     +-----+   verify_grounding   +-----+
           |                     |     +----------------------+     |
           |            [Hallucination]                       [Grounded]
           |                     |                                  |
           |                     v                                  v
           |          +---------------------+                      END
           +--------->| handle_hallucination|
                       +----------+----------+
                                  |
                                  v
                                 END
```

A fuller step-by-step lifecycle is in `docs/FLOW.md`; topology and latency budgets in `docs/ARCHITECTURE.md`; the reasoning behind the design choices in `docs/DECISIONS.md` (ADRs).

---

## 📦 Project Directory Structure

```
financial-intelligence-rag/
├── .github/workflows/ci.yml       # GitHub Actions test & build workflow
├── benchmarks/
│   ├── benchmark_retrieval.py     # Retrieval-latency benchmark (real runs only)
│   └── README.md                  # What the benchmark does / does not measure
├── data/
│   ├── README.md                  # Data layout and how to fetch filings
│   └── raw/                       # Staging directory for SEC PDFs (EDGAR lands here)
├── scripts/
│   ├── fetch_filing.py            # Download the latest filing for a ticker from EDGAR
│   └── make_sample_filing.py      # Generate a tiny synthetic sample filing PDF
├── docs/
│   ├── ARCHITECTURE.md            # System topology & latency budgets
│   ├── CONSTRAINTS.md             # Non-negotiable technical guardrails
│   ├── DECISIONS.md               # Architecture Decision Records (ADRs)
│   ├── FLOW.md                    # Detailed step-by-step query lifecycle
│   └── HANDOVER.md                # Engineering session handoff summary
├── src/
│   ├── app.py                     # FastAPI server (health, ingest, ingest-edgar, query)
│   ├── config.py                  # Pydantic Settings & environment schema
│   ├── database.py                # Qdrant manager with auto payload indexing; search_filtered
│   ├── edgar.py                   # SEC EDGAR client (ticker -> CIK -> filing)
│   ├── graph.py                   # Compiled LangGraph state machine & edges
│   ├── nodes.py                   # Graph nodes: decompose, retrieve+rerank, grade, generate, verify
│   ├── parser.py                  # PDF table extraction + prose chunking (1200/150)
│   └── state.py                   # TypedDict state definition
├── tests/
│   ├── test_pipeline.py           # End-to-end pipeline tests (mocked graph)
│   ├── test_nodes.py              # Unit tests for individual graph nodes
│   ├── test_chunking.py           # Prose chunker + parse_pdf tests
│   ├── test_metadata_isolation.py # search_filtered always filters ticker/year/period
│   └── test_edgar.py              # EDGAR client tests (mocked HTTP, no network)
├── .env.example                   # Environment configuration template
├── docker-compose.yml             # Multi-service Docker deployment config
├── Dockerfile                     # API container image (Python 3.11)
├── Dockerfile.frontend            # Streamlit frontend container image
├── frontend.py                    # Streamlit interactive web dashboard
├── LICENSE                        # MIT License
└── requirements.txt
```

---

## ⚡ Full Execution & Deployment Guide

### 1. Setup

```bash
git clone https://github.com/ishaaaan17/financial-intelligence-rag.git
cd financial-intelligence-rag

python -m venv rag
.\rag\Scripts\Activate.ps1       # Windows PowerShell
source rag/bin/activate          # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key_here
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=sec_filings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
```

### 2. Ingest a filing

```bash
python scripts/fetch_filing.py --ticker AAPL --form 10-Q   # from EDGAR, into data/raw/
# then upload it via POST /ingest with its ticker / fiscal_year / fiscal_period,
# or POST /ingest-edgar { "ticker": "AAPL", "form": "10-Q" } to fetch it server-side
```

### 3. Run the application

**Local (two terminals):**

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000   # API + Swagger at /docs
streamlit run frontend.py                                   # UI at http://localhost:8501
```

**Docker:**

```bash
docker-compose up --build -d
docker-compose logs -f
```

### 4. Run the tests

```bash
pytest -v tests/
```

37 tests; none of them need network access or API keys (EDGAR and the graph are mocked where appropriate).

### 5. Measure retrieval (optional, needs a populated Qdrant)

```bash
python benchmarks/benchmark_retrieval.py
```

Read `benchmarks/README.md` first — it states precisely what the script times (filtered vector search only) and what it does not (end-to-end query latency).

---

## 📡 REST API Reference

### `GET /health`
Probe endpoint for cloud health checks and load balancers.

### `POST /ingest`
Upload and tag an SEC 10-Q/10-K filing PDF.
**Form Data:** `ticker` (str), `fiscal_year` (int), `fiscal_period` (str), `file` (PDF binary).
Storage failures return a real 500 — a 200 means the filing was actually written to Qdrant.

### `POST /ingest-edgar`
Fetch the latest filing of a given form for a ticker directly from SEC EDGAR and save its primary document into `data/raw/`.
**JSON Body:** `{ "ticker": "AAPL", "form": "10-Q" }`

### `POST /query`
Execute a multi-period comparative financial query.

**Payload:**
```json
{ "query": "Compare Apple's R&D spend between Q2 2024 and Q2 2025" }
```

**Response:**
```json
{
  "query": "Compare Apple's R&D spend between Q2 2024 and Q2 2025",
  "ticker": "AAPL",
  "answer": "### Comparison of Apple's R&D Spend...\n\n| Period | R&D Spend (in millions) | Absolute Change | Percentage Change |\n| --- | --- | --- | --- |\n| Q2 2024 | $7,907 | - | - |\n| Q2 2025 | $8,550 | +$643 | +8.13% |",
  "citations": [
    { "ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "Q2", "page": 1, "type": "table" }
  ],
  "hallucination_detected": false
}
```

---

## 🌐 Hosting

The previous AWS EC2 demo instance is no longer running, so its URLs have been removed rather than left as dead links. The current hosting plan:

| Piece | Host |
|---|---|
| **Frontend** (web UI) | **Vercel** — static UI (built separately) |
| **Backend API** (FastAPI, this repo) | **Hugging Face Spaces (Docker)** or **Render** — this repo's `Dockerfile` |
| **Vector store** | Qdrant Cloud free tier (unchanged) |
| **LLM** | Groq API (unchanged) |

Deployment URLs will be added here once the frontend and API are live. Until then, run everything locally with the guide above — Docker (`docker-compose up`) works exactly as documented.

---

## Known Limitations

- **Not currently deployed.** The EC2 demo is gone and the Vercel + HF Spaces/Render hosting above is a plan with the pieces built, not a live deployment. There is no public URL for this system right now.
- **Latency figures are budgets, not measurements.** The ~2.5–4.5 s end-to-end figure is the design budget in `docs/ARCHITECTURE.md`; retrieval latency has a benchmark script but has not been run against a populated production collection, so no measured number is quoted.
- **No accuracy evaluation exists.** There is no labelled question/answer set for this corpus, so retrieval quality and answer correctness are not quantified. The grounding verifier checks each answer's numbers against its own retrieved context — it can catch numbers that appear nowhere in the filing text, but it cannot catch a *wrong period's* real number if decomposition targeted the wrong period, an error inside the filing itself, or a subtly wrong non-numerical claim. It is an LLM judgement, not a proof.
- **Decomposition is LLM-driven.** An ambiguously worded question can be decomposed into the wrong periods or metric; the grading and missing-data paths keep that from becoming a silently invented answer, but they cannot make the interpretation right.
- **Free-tier ceilings.** Groq's free tier rate-limits synthesis (the dominant latency cost), and Qdrant's free tier caps collection size; a large multi-company corpus would need paid tiers or self-hosting.
- **Ingestion is not query-fast.** Parsing and embedding a full 10-K takes seconds to ~a minute depending on length and table density; it runs outside the query path by design.

## License

Distributed under the MIT License. See `LICENSE` for details.
