"""
Streamlit Web UI for Financial Intelligence RAG Engine.
Provides interactive filing ingestion, natural language querying, metrics visualization, and citations.
"""

import os

import streamlit as st
import httpx
import pandas as pd

# Defaults to localhost for local dev; override with API_BASE_URL env var
# (e.g. http://financial-rag-api:8000 when running via docker-compose).
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="SEC Financial Intelligence Engine",
    page_icon="📈",
    layout="wide"
)

# --- Theme: matches the backend dashboard (ink / amber / filing-cover-sheet) ---
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
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
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 15% -10%, rgba(232,163,61,0.06), transparent 60%),
            var(--ink);
        color: var(--text);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--hairline);
    }

    /* Eyebrow + serif title, same voice as the backend cover sheet */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--amber);
        margin: 0 0 0.4rem;
    }
    .page-title {
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-weight: 600;
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 0.5rem;
        color: var(--text);
    }
    .page-sub {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0 0 1.75rem;
        max-width: 62ch;
    }

    /* Status card in sidebar, mirrors the backend status badge */
    .status-card {
        border: 1px solid var(--hairline);
        background: var(--panel-2);
        border-radius: 3px;
        padding: 0.85rem 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }
    .status-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
    .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .dot.live { background: var(--green); box-shadow: 0 0 0 3px rgba(74,222,128,0.15); }
    .dot.down { background: var(--red); box-shadow: 0 0 0 3px rgba(243,107,107,0.15); }
    .status-label { color: var(--text); }
    .status-field { color: var(--muted); font-size: 0.74rem; }
    .status-field b { color: var(--text); font-weight: 500; }

    /* Sidebar section labels */
    .sidebar-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.5rem 0 0.75rem;
    }

    /* Verdict banners for the grounding check, replacing default st.success/error */
    .verdict {
        border: 1px solid var(--hairline);
        border-left: 3px solid var(--muted);
        background: var(--panel);
        padding: 0.85rem 1.1rem;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        margin-bottom: 1.25rem;
    }
    .verdict.ok { border-left-color: var(--green); color: var(--text); }
    .verdict.ok b { color: var(--green); }
    .verdict.bad { border-left-color: var(--red); color: var(--text); }
    .verdict.bad b { color: var(--red); }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.25rem 0 0.75rem;
    }

    /* Answer panel */
    .answer-panel {
        border: 1px solid var(--hairline);
        background: var(--panel);
        padding: 1.5rem 1.75rem;
        border-radius: 3px;
        margin-bottom: 1.5rem;
    }
    .answer-panel table { font-size: 0.9rem; }
    .answer-panel code { color: var(--amber); background: var(--panel-2); }

    /* Inputs */
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
        background: var(--panel-2) !important;
        border: 1px solid var(--hairline) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
        color: var(--muted) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Primary button styled as a filing stamp */
    .stButton button, .stFormSubmitButton button {
        background: var(--amber-dim) !important;
        border: 1px solid var(--amber) !important;
        color: var(--amber) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        border-radius: 3px !important;
        transition: background 0.15s ease, color 0.15s ease;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        background: var(--amber) !important;
        color: var(--ink) !important;
    }

    .streamlit-expanderHeader {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
        color: var(--text) !important;
        background: var(--panel) !important;
        border: 1px solid var(--hairline) !important;
    }

    hr, div[data-testid="stSidebar"] hr { border-color: var(--hairline); }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="eyebrow">Console · Multi-Hop Analysis</p>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">SEC Financial Intelligence RAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Query, compare, and extract verified metrics from SEC 10-Q and 10-K filings. '
    'Every figure is checked against raw filing context before it reaches you.</p>',
    unsafe_allow_html=True
)

# Sidebar: Health Status & Ingestion Form
with st.sidebar:
    st.markdown('<p class="sidebar-eyebrow">System status</p>', unsafe_allow_html=True)

    try:
        health_resp = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.markdown(f"""
            <div class="status-card">
                <div class="status-row">
                    <span class="dot live"></span>
                    <span class="status-label">Backend online</span>
                </div>
                <div class="status-field">LLM &nbsp;<b>{data.get('llm_model')}</b></div>
                <div class="status-field">Collection &nbsp;<b>{data.get('collection')}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <div class="status-row"><span class="dot down"></span><span class="status-label">Backend error</span></div>
                <div class="status-field">Health probe returned a non-200 status.</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class="status-card">
            <div class="status-row"><span class="dot down"></span><span class="status-label">Backend offline</span></div>
            <div class="status-field">Start it with <b>uvicorn src.app:app</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-eyebrow">Upload SEC filing</p>', unsafe_allow_html=True)
    with st.form("ingest_form", clear_on_submit=True):
        ticker_input = st.text_input("Ticker symbol", value="AAPL").upper()
        col_y, col_p = st.columns(2)
        with col_y:
            year_input = st.number_input("Fiscal year", min_value=2015, max_value=2030, value=2024)
        with col_p:
            period_input = st.selectbox("Fiscal period", options=["Q1", "Q2", "Q3", "Q4", "FY"])

        uploaded_pdf = st.file_uploader("Filing PDF", type=["pdf"])
        submit_upload = st.form_submit_button("Ingest to Qdrant")

        if submit_upload and uploaded_pdf:
            with st.spinner("Extracting tables and embedding vectors…"):
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                data = {
                    "ticker": ticker_input,
                    "fiscal_year": str(year_input),
                    "fiscal_period": period_input
                }
                try:
                    resp = httpx.post(f"{API_BASE_URL}/ingest", data=data, files=files, timeout=60.0)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        st.success(f"Stored {res_json.get('chunks_stored')} chunks for {ticker_input} {period_input} {year_input}.")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to server: {str(e)}")

# Main Area: Query Interface
st.markdown('<p class="section-label">Query</p>', unsafe_allow_html=True)
query_example = "Compare Apple's R&D spend between Q2 2024 and Q2 2025"
user_query = st.text_input("Natural language financial query", value=query_example, label_visibility="collapsed")

if st.button("Run multi-hop analysis", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Decomposing query, retrieving filtered SEC chunks, and verifying grounding…"):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/query",
                    json={"query": user_query},
                    timeout=60.0
                )
                if response.status_code == 200:
                    result = response.json()

                    # 1. Verdict banner
                    if result.get("hallucination_detected"):
                        st.markdown(
                            '<div class="verdict bad"><b>Hallucination alert</b> — verification detected '
                            'ungrounded data. Response restricted.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="verdict ok"><b>Grounding verified</b> — every figure matched raw filing context.</div>',
                            unsafe_allow_html=True
                        )

                    # 2. Answer
                    st.markdown('<p class="section-label">Analysis & comparison</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-panel">{result.get("answer", "")}</div>', unsafe_allow_html=True)

                    # 3. Citations & provenance
                    citations = result.get("citations", [])
                    if citations:
                        with st.expander(f"Provenance citations ({len(citations)} chunks used)", expanded=False):
                            df_citations = pd.DataFrame(citations)
                            st.dataframe(df_citations, use_container_width=True)
                else:
                    st.error(f"Query failed ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API server: {str(e)}")