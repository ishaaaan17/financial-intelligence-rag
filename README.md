# 📈 SEC Multi-Hop Financial Intelligence RAG Engine

[![CI Pipeline](https://github.com/ishaaaan17/financial-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ishaaaan17/financial-intelligence-rag/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](<https://img.shields.io/badge/LLM-Groq%20(Llama--3.3--70B)-F05A28.svg>)](https://groq.com)
[![Docker](https://img.shields.io/badge/Deployed%20with-Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![AWS EC2](https://img.shields.io/badge/Hosted%20on-AWS%20EC2-FF9900.svg?logo=amazonaws&logoColor=white)](https://aws.amazon.com/ec2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, self-correcting Retrieval-Augmented Generation (RAG) agent engineered to perform complex, multi-period comparative financial analysis across SEC 10-Q and 10-K filings with zero hallucination.

---

## 🌐 Live Deployment

Both services are containerized with Docker and deployed on an **AWS EC2** instance.

| Service                                    | URL                                                      |
| ------------------------------------------ | -------------------------------------------------------- |
| 🔌 **Backend API** (FastAPI, Swagger docs) | [http://13.51.255.255:8000/](http://13.51.255.255:8000/) |
| 🖥️ **Frontend** (Streamlit dashboard)      | [http://13.51.255.255:8501/](http://13.51.255.255:8501/) |

> Note: this is a demo deployment on a plain HTTP EC2 endpoint (no domain/TLS yet), so browsers may show a "not secure" warning — that's expected.

---

## 🌟 Key Architecture Highlights

- **Multi-Hop Query Planning:** Deconstructs comparative user prompts (e.g., _"Compare Apple's R&D spend between Q2 2024 and Q2 2025"_) into deterministic metadata targets (`ticker`, `years`, `quarters`, `metric`).
- **Metadata-Isolated Vector Search:** Enforces strict payload filters (`ticker`, `fiscal_year`, `fiscal_period`) inside Qdrant Cloud to prevent cross-quarter context contamination.
- **Table-Preserving Ingestion Pipeline:** Extracts financial tables from SEC PDFs using `pdfplumber` and serializes them into structured Markdown tables to maintain numerical column alignments.
- **FlashRank Cross-Encoder Reranking:** Applies ultra-fast, local cross-encoder reranking (`ms-marco-TinyBERT-L-2-v2`) to prioritize relevant financial tables over boilerplate disclosure narrative.
- **Self-Correcting LangGraph State Machine:** Features conditional routing nodes:
  - `grade_documents`: Fallback to safe missing-period notifications if data is missing.
  - `verify_grounding`: Cross-references generated numerical claims against raw filing context to detect and block hallucinations.
- **Zero-Dollar Cloud Stack:** Powered by local HuggingFace embeddings (`all-MiniLM-L6-v2`), Qdrant Cloud Free Tier, and Groq's free API tier.
- **Containerized AWS Deployment:** Backend and frontend run as separate Docker containers (via `docker-compose`) on a single AWS EC2 instance, with ports 8000 (API) and 8501 (UI) exposed through the instance's security group.

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
├── docker-compose.yml             # Multi-service AWS EC2 deployment config
├── Dockerfile                     # Optimized Python 3.11 container image
├── frontend.py                    # Streamlit interactive web dashboard
├── README.md                      # Project documentation
└── requirements.txt                # Pinned Python package dependencies
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

### Option B: Docker Deployment (used in production on AWS EC2)

```bash
# Build and start container
docker-compose up --build -d

# Check running logs
docker-compose logs -f
```

On the EC2 instance, the same commands are run after opening inbound TCP access to ports `8000` and `8501` in the instance's security group, so the services are reachable at the instance's public IP (see **Live Deployment** above).

---

## 🧪 Running Automated Tests

Run the full test suite using pytest:

```bash
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
