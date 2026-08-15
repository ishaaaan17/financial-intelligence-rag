"""
Automated unit and integration test suite.
Validates parser markdown table formatting, config validation, and API routes,
including the graph's missing-data and hallucination-fallback branches.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from src.config import settings
from src.parser import SECParser
from src.app import app, financial_rag_app

client = TestClient(app)


def test_config_defaults():
    """Validates that base system defaults and models load properly."""
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.llm_temperature == 0.0
    assert settings.collection_name == "sec_filings"


def test_parser_table_to_markdown():
    """Validates that raw table grids convert into formatted markdown tables."""
    parser = SECParser(ticker="AAPL", fiscal_year=2024, fiscal_period="Q2")
    raw_table = [
        ["Line Item", "Three Months Ended March 30, 2024", "Three Months Ended April 1, 2023"],
        ["Research and development", "$7,907", "$7,457"],
        ["Total operating expenses", "$14,371", "$13,658"]
    ]

    md_output = parser._table_to_markdown(raw_table)

    assert "| Line Item | Three Months Ended March 30, 2024 | Three Months Ended April 1, 2023 |" in md_output
    assert "| --- | --- | --- |" in md_output
    assert "| Research and development | $7,907 | $7,457 |" in md_output


def test_health_endpoint():
    """Validates the /health endpoint response code and payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["collection"] == "sec_filings"


def test_query_schema_validation():
    """Validates query format rejection when payload is invalid."""
    response = client.post("/query", json={})
    assert response.status_code == 422


# --- /query: graph outcome paths -------------------------------------------
# These mock financial_rag_app.invoke directly, so they test the API's handling
# of each graph outcome shape without running the real LangGraph pipeline
# (no Groq/Qdrant/HuggingFace calls).

def test_query_returns_missing_data_response(monkeypatch):
    """When the graph routes through handle_missing_data, /query should surface
    the safe 'insufficient data' answer with empty citations and no hallucination flag."""
    fake_final_state = {
        "ticker": "AAPL",
        "final_answer": (
            "Insufficient data in vector store for AAPL. Could not find verified filing "
            "chunks for periods: Q2 2025. Please ensure the corresponding SEC 10-Q/10-K "
            "PDFs have been ingested."
        ),
        "citations": [],
        "hallucination_detected": False,
    }
    monkeypatch.setattr(financial_rag_app, "invoke", MagicMock(return_value=fake_final_state))

    response = client.post("/query", json={"query": "Compare AAPL R&D Q2 2024 vs Q2 2025"})

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["hallucination_detected"] is False
    assert data["citations"] == []
    assert "Insufficient data" in data["answer"]


def test_query_returns_hallucination_fallback_response(monkeypatch):
    """When the graph exhausts its retry budget in handle_hallucination, /query
    should surface the safety-warning fallback with hallucination_detected=True,
    never the ungrounded generated answer."""
    fake_final_state = {
        "ticker": "AAPL",
        "final_answer": (
            "Safety Warning: The generated comparative metrics could not be verified "
            "against the raw SEC source tables after 2 attempt(s). Generation blocked "
            "to maintain zero-hallucination policy."
        ),
        "citations": [],
        "hallucination_detected": True,
    }
    monkeypatch.setattr(financial_rag_app, "invoke", MagicMock(return_value=fake_final_state))

    response = client.post("/query", json={"query": "Compare AAPL R&D Q2 2024 vs Q2 2025"})

    assert response.status_code == 200
    data = response.json()
    assert data["hallucination_detected"] is True
    assert "Safety Warning" in data["answer"]


def test_query_returns_grounded_success_response(monkeypatch):
    """Happy path: grounded answer with citations flows through unchanged."""
    fake_final_state = {
        "ticker": "AAPL",
        "final_answer": "| Metric | Q2 2024 | Q2 2025 |\n| --- | --- | --- |\n| R&D | $7,907M | $8,200M |",
        "citations": [
            {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "Q2", "page": 3, "type": "table"}
        ],
        "hallucination_detected": False,
    }
    monkeypatch.setattr(financial_rag_app, "invoke", MagicMock(return_value=fake_final_state))

    response = client.post("/query", json={"query": "Compare AAPL R&D Q2 2024 vs Q2 2025"})

    assert response.status_code == 200
    data = response.json()
    assert data["hallucination_detected"] is False
    assert len(data["citations"]) == 1
    assert "R&D" in data["answer"]


def test_query_handles_missing_final_answer_gracefully(monkeypatch):
    """If the graph somehow returns no final_answer, /query should fall back to
    the default message rather than returning null/erroring (see app.py's
    final_output.get('final_answer', 'Unable to process query.'))."""
    fake_final_state = {
        "ticker": None,
        "citations": [],
        "hallucination_detected": False,
    }
    monkeypatch.setattr(financial_rag_app, "invoke", MagicMock(return_value=fake_final_state))

    response = client.post("/query", json={"query": "Something malformed"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Unable to process query."