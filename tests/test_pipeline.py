"""
Automated unit and integration test suite.
Validates parser markdown table formatting, config validation, and API routes.
"""

import pytest
from fastapi.testclient import TestClient
from src.config import settings
from src.parser import SECParser
from src.app import app

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