"""
Metadata-isolation tests for VectorStoreManager.search_filtered.

The project's central correctness claim is that retrieval for one
(ticker, fiscal_year, fiscal_period) can never return chunks from another.
That isolation lives entirely in the Qdrant Filter that search_filtered
builds, so these tests mock the client + embeddings and assert on the
exact filter that would be sent to Qdrant.

Lightweight by design: no Qdrant server, no embedding model download.
"""

import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")

# langchain_huggingface pulls in sentence-transformers/torch; it is only
# needed when embeddings are actually constructed, which these tests mock.
# Stub it only if it is not installed (e.g. a minimal local test env).
try:
    import langchain_huggingface  # noqa: F401
except ImportError:  # pragma: no cover - depends on environment
    sys.modules["langchain_huggingface"] = MagicMock()

from src.database import VectorStoreManager  # noqa: E402


def _manager_with_mock_client(payload):
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.embeddings = MagicMock()
    manager.embeddings.embed_query.return_value = [0.1] * 384
    manager.collection_name = "sec_filings"

    point = MagicMock()
    point.payload = payload
    response = MagicMock()
    response.points = [point]
    manager.client = MagicMock()
    manager.client.query_points.return_value = response
    return manager


def _filter_conditions(manager):
    _, kwargs = manager.client.query_points.call_args
    query_filter = kwargs["query_filter"]
    return {cond.key: cond.match.value for cond in query_filter.must}


def test_search_always_filters_on_ticker_year_and_period():
    manager = _manager_with_mock_client(
        {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q2",
         "page": 3, "type": "table", "page_content": "| R&D | $8,550 |"}
    )

    docs = manager.search_filtered(
        query="R&D spend", ticker="aapl", fiscal_year=2025, fiscal_period="q2", limit=5
    )

    conditions = _filter_conditions(manager)
    # All three isolation keys are present, normalised exactly as the
    # parser stores them (ticker/period upper-cased, year as int).
    assert conditions == {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q2"}
    # And the returned document carries content + metadata, not the raw payload.
    assert docs[0].page_content == "| R&D | $8,550 |"
    assert docs[0].metadata["ticker"] == "AAPL"
    assert "page_content" not in docs[0].metadata


def test_filter_changes_with_the_requested_period():
    """Two different periods must produce two different filters — a filter
    that ignored its arguments would silently mix quarters."""
    payload = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "Q2", "page_content": "x"}

    manager_q2_2024 = _manager_with_mock_client(payload)
    manager_q2_2024.search_filtered("rev", "AAPL", 2024, "Q2")

    manager_q3_2025 = _manager_with_mock_client(payload)
    manager_q3_2025.search_filtered("rev", "AAPL", 2025, "Q3")

    assert _filter_conditions(manager_q2_2024)["fiscal_period"] == "Q2"
    assert _filter_conditions(manager_q2_2024)["fiscal_year"] == 2024
    assert _filter_conditions(manager_q3_2025)["fiscal_period"] == "Q3"
    assert _filter_conditions(manager_q3_2025)["fiscal_year"] == 2025
