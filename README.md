# 📈 SEC Multi-Hop Financial Intelligence RAG Engine

[![CI Pipeline](https://github.com/ishaaaan17/financial-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ishaaaan17/financial-intelligence-rag/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](<https://img.shields.io/badge/LLM-Groq%20(Llama--3.3--70B)-F05A28.svg>)](https://groq.com)
[![Docker](https://img.shields.io/badge/Deployed%20with-Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, self-correcting Retrieval-Augmented Generation (RAG) agent engineered to perform complex, multi-period comparative financial analysis across SEC 10-Q and 10-K filings, with an LLM-based grounding-verification step that checks generated claims against the retrieved filing text before answering.

---

## 🌐 Hosting

The previous AWS EC2 demo instance is no longer running, so its URLs have
been removed rather than left as dead links. The current hosting plan:

| Piece                                   | Host                                                                 |
| --------------------------------------- | -------------------------------------------------------------------- |
| 🖥️ **Frontend** (web UI)                 | **Vercel** — static UI (built separately, in `financial-rag-vercel/`) |
| 🔌 **Backend API** (FastAPI, this repo)  | **Hugging Face Spaces (Docker)** or **Render** — use this repo's `Dockerfile` |
| 🗄️ **Vector store**                      | Qdrant Cloud free tier (unchanged)                                    |
| 🧠 **LLM**                               | Groq API (unchanged)                                                  |

> Deployment URLs will be added here once the Vercel frontend and the API
> space are live. Until then, run everything locally with the Quick Start
> below — Docker (`docker-compose up`) still works exactly as before.
> End-to-end query latency is budgeted at ~2.5–4.5 s per comparison
> (see `docs/ARCHITECTURE.md` §4); retrieval-only latency can be measured
> with `benchmarks/benchmark_retrieval.py`.

---

## 🌟 Key Architecture Highlights

- **Multi-Hop Query Planning:** Deconstructs comparative user prompts (e.g., _"Compare Apple's R&D spend between Q2 2024 and Q2 2025"_) into deterministic metadata targets (`ticker`, `years`, `quarters`, `metric`).
- **Metadata-Isolated Vector Search:** Enforces strict payload filters (`ticker`, `fiscal_year`, `fiscal_period`) inside Qdrant Cloud to prevent cross-quarter context contamination.
- **Table-Preserving Ingestion Pipeline:** Extracts financial tables from SEC PDFs using `pdfplumber` and serializes them into structured Markdown tables to maintain numerical column alignments.
- **FlashRank Cross-Encoder Reranking:** Applies ultra-fast, local cross-encoder reranking (`ms-marco-TinyBERT-L-2-v2`) inside the retrieval step — the 5 chunks retrieved per period are reranked down to the top 3 — to prioritize relevant financial tables over boilerplate disclosure narrative.
- **SEC EDGAR Ingestion:** Fetches real filings directly from EDGAR (`src/edgar.py`, `POST /ingest-edgar`, or `python scripts/fetch_filing.py --ticker AAPL`) instead of relying only on manual PDF uploads.
- **Self-Correcting LangGraph State Machine:** Features conditional routing nodes:
  - `grade_documents`: Fallback to safe missing-period notifications if data is missing.
  - `verify_grounding`: Cross-references generated numerical claims against raw filing context to detect and block hallucinations.
- **Zero-Dollar Cloud Stack:** Powered by local HuggingFace embeddings (`all-MiniLM-L6-v2`), Qdrant Cloud Free Tier, and Groq's free API tier.
- **Containerized Deployment:** Backend and frontend run as separate Docker containers (via `docker-compose`); the API image is the same one used for the Hugging Face Spaces / Render deployment (see **Hosting** above).

---

## 🏗️ System Architecture & Workflow

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
            +----------------------------+
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

---

## 📁 Repository Structure

```
financial-intelligence-rag/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions automated test & build workflow
├── benchmarks/
│   ├── benchmark_retrieval.py     # Retrieval-latency benchmark (real runs only)
│   └── README.md                  # What the benchmark does / does not measure
├── data/
│   ├── README.md                  # Data layout and how to fetch filings
│   └── raw/                       # Staging directory for sample SEC PDFs
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
│   ├── __init__.py
│   ├── app.py                     # FastAPI server with custom UI & CORS
│   ├── config.py                  # Pydantic Settings & environment schema
│   ├── database.py                # Qdrant manager with auto-payload indexing
│   ├── edgar.py                   # SEC EDGAR client (ticker -> CIK -> filing)
│   ├── graph.py                   # Compiled LangGraph state machine & edges
│   ├── nodes.py                   # Graph nodes, reranking, & verification
│   ├── parser.py                  # SEC PDF & Markdown table extraction + chunking
│   └── state.py                   # TypedDict state definition
├── tests/
│   ├── test_pipeline.py           # End-to-end pipeline tests (mocked graph)
│   ├── test_nodes.py              # Unit tests for individual graph nodes
│   ├── test_chunking.py           # Prose chunker + parse_pdf tests
│   ├── test_metadata_isolation.py # search_filtered always filters ticker/year/period
│   └── test_edgar.py              # EDGAR client tests (mocked HTTP, no network)
├── .env.example                   # Environment configuration template
├── .gitignore
├── docker-compose.yml             # Multi-service Docker deployment config
├── Dockerfile                     # Optimized Python 3.11 container image
├── Dockerfile.frontend            # Streamlit frontend container image
├── frontend.py                    # Streamlit interactive web dashboard
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
└── requirements.txt                # Python package dependencies (minimum versions)
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository & Setup Environment

```bash
git clone https://github.com/ishaaaan17/financial-intelligence-rag.git
cd financial-intelligence-rag

# Create and activate virtual environment
python -m venv rag

# Windows PowerShell:
.\rag\Scripts\Activate.ps1

# Linux / macOS:
source rag/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key_here
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=sec_filings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
```

---

## 💻 Running the Application

### Option A: Local Development

Open two terminal tabs:

**Terminal 1 — FastAPI Backend:**

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

- Root Admin Dashboard: <http://127.0.0.1:8000/>
- Interactive Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc Viewer: <http://127.0.0.1:8000/redoc>

**Terminal 2 — Streamlit Frontend:**

```bash
streamlit run frontend.py
```

- Streamlit Web UI: <http://localhost:8501>

### Option B: Docker Deployment

```bash
# Build and start container
docker-compose up --build -d

# Check running logs
docker-compose logs -f
```

The same API image is used for the hosted deployment (see **Hosting** above).

---

## 🧪 Running Automated Tests

Run the full test suite using pytest:

```bash
pytest -v tests/
```

---

## 📡 REST API Reference

### `GET /health`

Probe endpoint for cloud health checks and load balancers.

### `POST /ingest`

Upload and tag an SEC 10-Q/10-K filing PDF.

**Form Data:** `ticker` (str), `fiscal_year` (int), `fiscal_period` (str), `file` (PDF binary).

### `POST /ingest-edgar`

Fetch the latest filing of a given form for a ticker directly from SEC
EDGAR and save its primary document into `data/raw/`.

**JSON Body:** `{ "ticker": "AAPL", "form": "10-Q" }` — also available as a
CLI: `python scripts/fetch_filing.py --ticker AAPL --form 10-Q`.

### `POST /query`

Execute a multi-period comparative financial query.

**Payload:**

```json
{
  "query": "Compare Apple's R&D spend between Q2 2024 and Q2 2025"
}
```

**Response:**

```json
{
  "query": "Compare Apple's R&D spend between Q2 2024 and Q2 2025",
  "ticker": "AAPL",
  "answer": "### Comparison of Apple's R&D Spend...\n\n| Period | R&D Spend (in millions) | Absolute Change | Percentage Change |\n| --- | --- | --- | --- |\n| Q2 2024 | $7,907 | - | - |\n| Q2 2025 | $8,550 | +$643 | +8.13% |",
  "citations": [
    {
      "ticker": "AAPL",
      "fiscal_year": 2024,
      "fiscal_period": "Q2",
      "page": 1,
      "type": "table"
    }
  ],
  "hallucination_detected": false
}
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
