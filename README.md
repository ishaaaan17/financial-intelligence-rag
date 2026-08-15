# 📈 SEC Multi-Hop Financial Intelligence RAG Engine

[![CI Pipeline](https://github.com/ishaaaan17/financial-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ishaaaan17/financial-intelligence-rag/actions) [![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com) [![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph) [![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech) [![Groq](<https://img.shields.io/badge/LLM-Groq%20(Llama--3.3--70B)-F05A28.svg>)](https://groq.com) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, self-correcting Retrieval-Augmented Generation (RAG) agent engineered to perform complex, multi-period comparative financial analysis across SEC 10-Q and 10-K filings with zero hallucination.

---

## 🌟 Key Architecture Highlights

- **Multi-Hop Query Planning:** Deconstructs comparative user prompts (e.g., _"Compare Apple's R&D spend between Q2 2024 and Q2 2025"_) into deterministic metadata targets (`ticker`, `years`, `quarters`, `metric`).
- **Metadata-Isolated Vector Search:** Enforces strict payload filters (`ticker`, `fiscal_year`, `fiscal_period`) inside Qdrant Cloud to prevent cross-quarter context contamination.
- **Table-Preserving Ingestion Pipeline:** Extracts financial tables from SEC PDFs using `pdfplumber` and serializes them into structured Markdown tables to maintain numerical column alignments.
- **FlashRank Cross-Encoder Reranking:** Applies ultra-fast, local cross-encoder reranking (`ms-marco-TinyBERT-L-2-v2`) to prioritize relevant financial tables over boilerplate disclosure narrative.
- **Self-Correcting LangGraph State Machine:** Features conditional routing nodes:
  - `grade_documents`: Fallback to safe missing-period notifications if data is missing.
  - `verify_grounding`: Cross-references generated numerical claims against raw filing context to detect hallucinations. On a failed check, the graph makes one bounded retry attempt (`increment_retry` → `generate_comparison`, capped by `MAX_GENERATION_RETRIES`) before falling back to a safe refusal via `handle_hallucination`.
- **Zero-Dollar Cloud Stack:** Powered by local HuggingFace embeddings (`all-MiniLM-L6-v2`), Qdrant Cloud Free Tier, and Groq's free API tier.

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
+-----------------------+       +-----------------------+
|  handle_missing_data  |       |  generate_comparison  |<--------+
+----------+------------+       +-----------+-----------+         |
           |                               |                     |
           |                               v                     |
           |                    +-----------------------+         |
           |                    |    verify_grounding     |         |
           |                    +-----------+-----------+         |
           |                       [Grounded]  |  [Hallucination]  |
           |                          |         |                 |
           |                          v         v                 |
           |                         END   retry budget left? -----+  (increment_retry,
           |                                    |                       loop back)
           |                                    | budget exhausted
           |                                    v
           |                        +----------------------+
           +----------------------->| handle_hallucination |
                                     +----------+-----------+
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
├── data/
│   └── raw/                       # Staging directory for sample SEC PDFs
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
│   ├── graph.py                   # Compiled LangGraph state machine & edges
│   ├── nodes.py                   # Graph nodes, reranking, & verification
│   ├── parser.py                  # SEC PDF & Markdown table extraction pipeline
│   └── state.py                   # TypedDict state definition
├── tests/
│   └── test_pipeline.py           # Automated unit and integration tests
├── .env.example                   # Environment configuration template
├── .gitignore
├── docker-compose.yml             # Multi-service local deployment config
├── Dockerfile                     # Optimized Python 3.11 container image
├── frontend.py                    # Streamlit interactive web dashboard
├── README.md                      # Project documentation
└── requirements.txt                # Pinned Python package dependencies
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository & Setup Environment

```
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

```
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the template and fill in real values:

```
cp .env.example .env
```

```
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
LLM_TIMEOUT_SECONDS=30.0
LLM_MAX_RETRIES=3
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=sec_filings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000
```

---

## 💻 Running the Application

### Option A: Local Development

Open two terminal tabs:

**Terminal 1 — FastAPI Backend:**

```
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

- Root Admin Dashboard: <http://127.0.0.1:8000/>
- Interactive Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc Viewer: <http://127.0.0.1:8000/redoc>

**Terminal 2 — Streamlit Frontend:**

```
streamlit run frontend.py
```

- Streamlit Web UI: <http://localhost:8501>

### Option B: Docker Deployment

```
# Build and start container
docker-compose up --build -d

# Check running logs
docker-compose logs -f
```

---

## 🧪 Running Automated Tests

Run the full test suite using pytest:

```
pytest -v tests/test_pipeline.py
```

---

## 📡 REST API Reference

### `GET /health`

Probe endpoint for cloud health checks and load balancers.

### `POST /ingest`

Upload and tag an SEC 10-Q/10-K filing PDF.

**Form Data:** `ticker` (str), `fiscal_year` (int), `fiscal_period` (str), `file` (PDF binary).

### `POST /query`

Execute a multi-period comparative financial query.

**Payload:**

```
{
  "query": "Compare Apple's R&D spend between Q2 2024 and Q2 2025"
}
```

**Response:**

```
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

`hallucination_detected` reflects the final outcome after the bounded retry loop — if `verify_grounding` fails, the graph retries synthesis once before either grounding successfully or falling back to a safe refusal (see `docs/ARCHITECTURE.md` §3 and `docs/CONSTRAINTS.md` §11 for the exact retry/backoff rules).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
