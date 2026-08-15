"""
FastAPI application interface.
Features rich interactive API documentation, CORS configuration,
health diagnostics, and vector collection metadata endpoints.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import tempfile

from src.config import settings
from src.parser import SECParser
from src.nodes import vector_store
from src.graph import financial_rag_app

# --- App Initialization & Custom Swagger Theme ---
app = FastAPI(
    title="⚡ Enterprise SEC Financial Intelligence Engine",
    description="""
### Multi-Hop SEC Filing Analysis & Grounded Comparison API

This backend orchestrates a **self-correcting LangGraph workflow** paired with **Qdrant Vector Search** and **Groq Llama 3.3-70B**.

---
* 🔍 **Smart Retrieval**: Period-filtered metadata vector searches with zero cross-quarter pollution.
* 🛡️ **Zero-Hallucination Guard**: Post-generation fact verification engine.
* 📊 **Table-Native Extraction**: Markdown table preserving parser for financial matrices.
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,       # Hides schema clutter at the bottom
        "docExpansion": "list",                # Clean listing of endpoints
        "filter": True,                        # Enables search filter bar
        "syntaxHighlight.theme": "monokai"     # Clean dark theme syntax for payloads
    }
)

# --- CORS configuration ---
# DEC-013: the wildcard origin was a tracked temporary shortcut for local dev, not a
# considered production choice. Origins now come from settings.allowed_origins
# (a comma-separated ALLOWED_ORIGINS env var, see config.py), defaulting to the
# local Streamlit + Swagger origins used in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# --- Pydantic Data Models with Rich OpenAPI Examples ---
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        title="Natural Language Financial Query",
        description="Comparative financial query across quarters or annual filings.",
        json_schema_extra={
            "example": "Compare Apple's R&D spend between Q2 2024 and Q2 2025"
        }
    )


class QueryResponse(BaseModel):
    query: str
    ticker: Optional[str]
    answer: str
    citations: List[Dict[str, Any]]
    hallucination_detected: bool


class IngestResponse(BaseModel):
    message: str
    ticker: str
    fiscal_year: int
    fiscal_period: str
    chunks_stored: int


# --- Root Dashboard (Rich Backend UI at http://127.0.0.1:8000/) ---
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
def backend_dashboard():
    """Renders the engine's status dashboard, styled as a filing cover sheet."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SEC Financial Intelligence Engine</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --ink: #0b0e14;
                --panel: #12161f;
                --panel-2: #171c27;
                --hairline: #262d3a;
                --text: #e7eaf2;
                --muted: #8992a6;
                --amber: #e8a33d;
                --amber-dim: rgba(232, 163, 61, 0.14);
                --green: #4ade80;
                --red: #f36b6b;
                --serif: 'IBM Plex Serif', Georgia, serif;
                --sans: 'IBM Plex Sans', -apple-system, sans-serif;
                --mono: 'IBM Plex Mono', 'Courier New', monospace;
            }}

            * {{ box-sizing: border-box; }}

            body {{
                margin: 0;
                background:
                    radial-gradient(1200px 500px at 15% -10%, rgba(232,163,61,0.06), transparent 60%),
                    var(--ink);
                color: var(--text);
                font-family: var(--sans);
                -webkit-font-smoothing: antialiased;
            }}

            a {{ color: inherit; }}

            /* --- Ticker tape: scrolling filing feed, the page's signature element --- */
            .ticker {{
                border-bottom: 1px solid var(--hairline);
                background: var(--panel);
                overflow: hidden;
                white-space: nowrap;
                position: relative;
            }}
            .ticker::before, .ticker::after {{
                content: "";
                position: absolute;
                top: 0; bottom: 0;
                width: 48px;
                z-index: 2;
                pointer-events: none;
            }}
            .ticker::before {{ left: 0; background: linear-gradient(90deg, var(--panel), transparent); }}
            .ticker::after {{ right: 0; background: linear-gradient(270deg, var(--panel), transparent); }}
            .ticker-track {{
                display: inline-flex;
                gap: 2.5rem;
                padding: 0.55rem 0;
                animation: scroll 32s linear infinite;
                font-family: var(--mono);
                font-size: 0.78rem;
                letter-spacing: 0.02em;
            }}
            .ticker-track span {{ color: var(--muted); }}
            .ticker-track b {{ color: var(--amber); font-weight: 600; }}
            @keyframes scroll {{
                from {{ transform: translateX(0); }}
                to {{ transform: translateX(-50%); }}
            }}
            @media (prefers-reduced-motion: reduce) {{
                .ticker-track {{ animation: none; }}
            }}

            .wrap {{
                max-width: 960px;
                margin: 0 auto;
                padding: 3rem 1.5rem 4rem;
            }}

            /* --- Header: filing cover sheet --- */
            header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1.5rem;
                border-bottom: 1px solid var(--hairline);
                padding-bottom: 1.75rem;
                margin-bottom: 2.25rem;
            }}
            .eyebrow {{
                font-family: var(--mono);
                font-size: 0.72rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: var(--amber);
                margin: 0 0 0.6rem;
            }}
            h1 {{
                font-family: var(--serif);
                font-weight: 600;
                font-size: 2rem;
                line-height: 1.15;
                margin: 0;
                max-width: 22ch;
            }}
            .status {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: var(--mono);
                font-size: 0.78rem;
                color: var(--muted);
                border: 1px solid var(--hairline);
                background: var(--panel);
                padding: 0.5rem 0.85rem;
                border-radius: 3px;
                white-space: nowrap;
            }}
            .dot {{
                width: 7px; height: 7px;
                border-radius: 50%;
                background: var(--muted);
                flex-shrink: 0;
            }}
            .dot.live {{ background: var(--green); box-shadow: 0 0 0 3px rgba(74,222,128,0.15); }}
            .dot.down {{ background: var(--red); box-shadow: 0 0 0 3px rgba(243,107,107,0.15); }}

            /* --- Spec sheet grid --- */
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1px;
                background: var(--hairline);
                border: 1px solid var(--hairline);
                margin-bottom: 1.75rem;
            }}
            @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
            .cell {{
                background: var(--panel);
                padding: 1.4rem 1.5rem;
            }}
            .cell h5 {{
                font-family: var(--sans);
                font-size: 0.72rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--muted);
                margin: 0 0 1rem;
                font-weight: 600;
            }}
            .field {{ margin-bottom: 0.9rem; }}
            .field:last-child {{ margin-bottom: 0; }}
            .field-label {{
                font-size: 0.72rem;
                color: var(--muted);
                margin-bottom: 0.2rem;
            }}
            .field-value {{
                font-family: var(--mono);
                font-size: 0.88rem;
                color: var(--text);
                background: var(--panel-2);
                border: 1px solid var(--hairline);
                border-radius: 3px;
                padding: 0.35rem 0.6rem;
                display: inline-block;
            }}

            /* --- Explorer panel --- */
            .panel {{
                border: 1px solid var(--hairline);
                background: var(--panel);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }}
            .panel h5 {{
                font-family: var(--sans);
                font-size: 0.72rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--muted);
                margin: 0 0 0.35rem;
                font-weight: 600;
            }}
            .panel p {{
                color: var(--muted);
                font-size: 0.88rem;
                margin: 0 0 1.1rem;
            }}
            .links {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
            }}
            .stamp {{
                font-family: var(--mono);
                font-size: 0.8rem;
                text-decoration: none;
                border: 1px solid var(--hairline);
                border-radius: 3px;
                padding: 0.6rem 1rem;
                color: var(--text);
                transition: border-color 0.15s ease, background 0.15s ease;
            }}
            .stamp:hover {{ border-color: var(--amber); background: var(--amber-dim); }}
            .stamp.primary {{
                border-color: var(--amber);
                background: var(--amber-dim);
                color: var(--amber);
            }}
            .stamp.primary:hover {{ background: var(--amber); color: var(--ink); }}

            footer {{
                font-family: var(--mono);
                font-size: 0.72rem;
                color: var(--muted);
                border-top: 1px solid var(--hairline);
                padding-top: 1.25rem;
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 0.5rem;
            }}

            :focus-visible {{
                outline: 2px solid var(--amber);
                outline-offset: 2px;
            }}
        </style>
    </head>
    <body>

        <div class="ticker" aria-hidden="true">
            <div class="ticker-track" id="tickerTrack"></div>
        </div>

        <div class="wrap">
            <header>
                <div>
                    <p class="eyebrow">Backend · Engine Status</p>
                    <h1>SEC Financial Intelligence Engine</h1>
                </div>
                <div class="status" id="statusBadge">
                    <span class="dot" id="statusDot"></span>
                    <span id="statusText">Checking…</span>
                </div>
            </header>

            <div class="grid">
                <div class="cell">
                    <h5>Inference</h5>
                    <div class="field">
                        <div class="field-label">Active model</div>
                        <div class="field-value">{settings.llm_model}</div>
                    </div>
                    <div class="field">
                        <div class="field-label">Temperature</div>
                        <div class="field-value">{settings.llm_temperature} · deterministic</div>
                    </div>
                </div>
                <div class="cell">
                    <h5>Vector store</h5>
                    <div class="field">
                        <div class="field-label">Collection</div>
                        <div class="field-value">{settings.collection_name}</div>
                    </div>
                    <div class="field">
                        <div class="field-label">Embeddings model</div>
                        <div class="field-value">{settings.embedding_model_name}</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <h5>Consoles</h5>
                <p>Query and ingestion interfaces for this engine.</p>
                <div class="links">
                    <a href="/docs" class="stamp primary">Swagger — /docs</a>
                    <a href="/redoc" class="stamp">ReDoc — /redoc</a>
                    <a href="http://localhost:8501" target="_blank" class="stamp">Streamlit UI</a>
                    <a href="/health" class="stamp">Health probe — /health</a>
                </div>
            </div>

            <footer>
                <span>financial-intelligence-rag · v1.1.0</span>
                <span id="latencyText">Ping —</span>
            </footer>
        </div>

        <script>
            // Decorative ticker: reads the same fields the cards show, twice, for a seamless loop.
            const items = [
                ["MODEL", "{settings.llm_model}"],
                ["COLLECTION", "{settings.collection_name}"],
                ["EMBEDDINGS", "{settings.embedding_model_name}"],
                ["TEMP", "{settings.llm_temperature}"],
                ["MODE", "MULTI-HOP · GROUNDED"],
            ];
            const track = document.getElementById('tickerTrack');
            const renderItems = () => items.map(([k, v]) => `<span>${{k}} <b>${{v}}</b></span>`).join('');
            track.innerHTML = renderItems() + renderItems();

            // Live health probe against the API itself.
            async function checkHealth() {{
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                const latency = document.getElementById('latencyText');
                const start = performance.now();
                try {{
                    const res = await fetch('/health');
                    const ms = Math.round(performance.now() - start);
                    if (res.ok) {{
                        dot.className = 'dot live';
                        text.textContent = 'Service active';
                        latency.textContent = `Ping ${{ms}}ms`;
                    }} else {{
                        throw new Error('non-200');
                    }}
                }} catch (e) {{
                    dot.className = 'dot down';
                    text.textContent = 'Service unreachable';
                    latency.textContent = 'Ping —';
                }}
            }}
            checkHealth();
            setInterval(checkHealth, 15000);
        </script>
    </body>
    </html>
    """


# --- Core API Endpoints ---
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Cloud probe health check endpoint."""
    return {
        "status": "healthy",
        "llm_model": settings.llm_model,
        "collection": settings.collection_name
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Data Pipeline"])
async def ingest_filing(
    ticker: str = Form(..., description="Stock symbol (e.g. AAPL, MSFT)"),
    fiscal_year: int = Form(..., description="Filing year (e.g. 2024, 2025)"),
    fiscal_period: str = Form(..., description="Period (Q1, Q2, Q3, Q4, or FY)"),
    file: UploadFile = File(..., description="SEC 10-Q/10-K PDF document")
):
    """
    Uploads an SEC filing PDF, extracts text and markdown tables,
    and upserts tagged vectors into Qdrant.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        parser = SECParser(ticker=ticker, fiscal_year=fiscal_year, fiscal_period=fiscal_period)
        chunks = parser.parse_pdf(tmp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text or tables found in PDF.")

        vector_store.insert_chunks(chunks)

        return IngestResponse(
            message="Filing successfully parsed, embedded, and stored in vector database.",
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period.upper(),
            chunks_stored=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/query", response_model=QueryResponse, tags=["Agent Execution"])
async def run_query(request: QueryRequest):
    """
    Executes the self-correcting LangGraph multi-hop analysis across SEC periods.
    """
    initial_state = {
        "original_query": request.query,
        "ticker": "",
        "target_periods": [],
        "metric_focus": "",
        "retrieved_docs": [],
        "missing_periods": [],
        "documents_are_relevant": False,
        "hallucination_detected": False,
        "retry_count": 0,
        "final_answer": None,
        "citations": []
    }

    final_output = financial_rag_app.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        ticker=final_output.get("ticker"),
        answer=final_output.get("final_answer", "Unable to process query."),
        citations=final_output.get("citations", []),
        hallucination_detected=final_output.get("hallucination_detected", False)
    )
