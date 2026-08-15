"""
Unit tests for individual LangGraph nodes (src/nodes.py) and the bounded
retry routing logic (src/graph.py).

These tests mock `src.nodes._invoke_chain` (the single choke point every node
routes LLM calls through) and the module-level `vector_store` / `ranker`
singletons, so no real Groq, Qdrant, or HuggingFace network calls are made.
Node business logic (period parsing, citation dedup, routing) is exercised
directly against controlled fake return values.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src import nodes
from src.nodes import QueryPlan, RelevanceGrade, GroundingVerification
from src.graph import (
    route_after_grading,
    route_after_verification,
    increment_retry,
    handle_hallucination,
    handle_missing_data,
    MAX_GENERATION_RETRIES,
)


def base_state(**overrides):
    """Minimal AgentState dict with sane defaults, overridable per test."""
    state = {
        "original_query": "Compare AAPL R&D spend Q2 2024 vs Q2 2025",
        "ticker": "AAPL",
        "target_periods": [],
        "metric_focus": "R&D expense",
        "retrieved_docs": [],
        "missing_periods": [],
        "documents_are_relevant": False,
        "hallucination_detected": False,
        "retry_count": 0,
        "final_answer": None,
        "citations": [],
    }
    state.update(overrides)
    return state


# --- decompose_query -------------------------------------------------------

def test_decompose_query_zips_matched_years_and_quarters(monkeypatch):
    fake_plan = QueryPlan(ticker="aapl", years=[2024, 2025], quarters=["q2", "q2"], metric_focus="R&D expense")
    mock_invoke = MagicMock(return_value=fake_plan)
    monkeypatch.setattr(nodes, "_invoke_chain", mock_invoke)

    result = nodes.decompose_query(base_state())

    assert result["ticker"] == "AAPL"
    assert result["target_periods"] == [
        {"year": 2024, "quarter": "Q2"},
        {"year": 2025, "quarter": "Q2"},
    ]
    assert result["retrieved_docs"] == []
    mock_invoke.assert_called_once()


def test_decompose_query_falls_back_to_cartesian_when_lengths_mismatch(monkeypatch):
    # 2 years, 1 quarter -> lengths don't match, so every year is paired with every quarter
    fake_plan = QueryPlan(ticker="msft", years=[2024, 2025], quarters=["q3"], metric_focus="Operating margin")
    monkeypatch.setattr(nodes, "_invoke_chain", MagicMock(return_value=fake_plan))

    result = nodes.decompose_query(base_state())

    assert result["target_periods"] == [
        {"year": 2024, "quarter": "Q3"},
        {"year": 2025, "quarter": "Q3"},
    ]


def test_decompose_query_preserves_existing_retry_count(monkeypatch):
    fake_plan = QueryPlan(ticker="aapl", years=[2024], quarters=["q1"], metric_focus="Revenue")
    monkeypatch.setattr(nodes, "_invoke_chain", MagicMock(return_value=fake_plan))

    result = nodes.decompose_query(base_state(retry_count=1))

    assert result["retry_count"] == 1


# --- grade_documents ---------------------------------------------------------

def test_grade_documents_short_circuits_on_missing_periods(monkeypatch):
    mock_invoke = MagicMock()
    monkeypatch.setattr(nodes, "_invoke_chain", mock_invoke)

    state = base_state(missing_periods=[{"year": 2024, "quarter": "Q2"}], retrieved_docs=[Document(page_content="x")])
    result = nodes.grade_documents(state)

    assert result == {"documents_are_relevant": False}
    mock_invoke.assert_not_called()


def test_grade_documents_short_circuits_on_empty_retrieved_docs(monkeypatch):
    mock_invoke = MagicMock()
    monkeypatch.setattr(nodes, "_invoke_chain", mock_invoke)

    state = base_state(missing_periods=[], retrieved_docs=[])
    result = nodes.grade_documents(state)

    assert result == {"documents_are_relevant": False}
    mock_invoke.assert_not_called()


def test_grade_documents_calls_llm_when_docs_present():
    monkeypatch = pytest.MonkeyPatch()
    try:
        fake_grade = RelevanceGrade(is_relevant=True)
        mock_invoke = MagicMock(return_value=fake_grade)
        monkeypatch.setattr(nodes, "_invoke_chain", mock_invoke)

        state = base_state(
            missing_periods=[],
            retrieved_docs=[Document(page_content="R&D was $7,907 million", metadata={})],
        )
        result = nodes.grade_documents(state)

        assert result == {"documents_are_relevant": True}
        mock_invoke.assert_called_once()
    finally:
        monkeypatch.undo()


# --- retrieve_filtered_docs --------------------------------------------------

def test_retrieve_filtered_docs_reranks_and_flags_missing_periods(monkeypatch):
    found_doc = Document(page_content="R&D $7,907M", metadata={"fiscal_period": "Q2", "fiscal_year": 2024})

    mock_search = MagicMock(side_effect=[[found_doc], []])  # first period found, second missing
    monkeypatch.setattr(nodes.vector_store, "search_filtered", mock_search)

    mock_rerank = MagicMock(return_value=[{"id": 0, "text": found_doc.page_content, "meta": found_doc.metadata}])
    monkeypatch.setattr(nodes.ranker, "rerank", mock_rerank)

    state = base_state(target_periods=[
        {"year": 2024, "quarter": "Q2"},
        {"year": 2025, "quarter": "Q2"},
    ])
    result = nodes.retrieve_filtered_docs(state)

    assert len(result["retrieved_docs"]) == 1
    assert result["missing_periods"] == [{"year": 2025, "quarter": "Q2"}]
    assert mock_search.call_count == 2
    mock_rerank.assert_called_once()


# --- generate_comparison ------------------------------------------------------

def test_generate_comparison_dedupes_citations_and_builds_context(monkeypatch):
    shared_meta = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "Q2", "page": 3, "type": "table"}
    doc_a = Document(page_content="R&D: $7,907M", metadata=shared_meta)
    doc_b = Document(page_content="R&D detail: $7,907M breakdown", metadata=shared_meta)  # same metadata as doc_a

    fake_response = SimpleNamespace(content="| Metric | Q2 2024 |\n| --- | --- |\n| R&D | $7,907M |")
    mock_invoke = MagicMock(return_value=fake_response)
    monkeypatch.setattr(nodes, "_invoke_chain", mock_invoke)

    state = base_state(retrieved_docs=[doc_a, doc_b])
    result = nodes.generate_comparison(state)

    assert result["final_answer"] == fake_response.content
    # Both docs share identical metadata -> should collapse to a single citation
    assert result["citations"] == [shared_meta]

    # Confirm both chunks made it into the context passed to the LLM
    _, payload = mock_invoke.call_args[0]
    assert "$7,907M" in payload["context"]
    assert "breakdown" in payload["context"]


# --- verify_grounding ----------------------------------------------------------

@pytest.mark.parametrize("is_grounded,expected_flag", [(True, False), (False, True)])
def test_verify_grounding_maps_to_hallucination_flag(monkeypatch, is_grounded, expected_flag):
    fake_verification = GroundingVerification(is_grounded=is_grounded, explanation="test")
    monkeypatch.setattr(nodes, "_invoke_chain", MagicMock(return_value=fake_verification))

    state = base_state(
        retrieved_docs=[Document(page_content="R&D was $7,907 million")],
        final_answer="R&D was $7,907 million",
    )
    result = nodes.verify_grounding(state)

    assert result == {"hallucination_detected": expected_flag}


# --- graph routing: bounded retry logic ----------------------------------------

def test_route_after_verification_ends_when_grounded():
    from langgraph.graph import END
    state = base_state(hallucination_detected=False)
    assert route_after_verification(state) == END


def test_route_after_verification_retries_when_budget_remains():
    state = base_state(hallucination_detected=True, retry_count=0)
    assert route_after_verification(state) == "increment_retry"


def test_route_after_verification_falls_back_when_budget_exhausted():
    state = base_state(hallucination_detected=True, retry_count=MAX_GENERATION_RETRIES)
    assert route_after_verification(state) == "handle_hallucination"


def test_increment_retry_bumps_counter():
    state = base_state(retry_count=0)
    result = increment_retry(state)
    assert result == {"retry_count": 1}


def test_handle_hallucination_reports_correct_attempt_count():
    # retry_count=1 means 1 retry already happened -> 2 total attempts were made
    state = base_state(retry_count=1)
    result = handle_hallucination(state)
    assert "2 attempt(s)" in result["final_answer"]


def test_route_after_grading_routes_to_missing_data_when_not_relevant():
    state = base_state(documents_are_relevant=False)
    assert route_after_grading(state) == "handle_missing_data"


def test_route_after_grading_routes_to_generation_when_relevant():
    state = base_state(documents_are_relevant=True)
    assert route_after_grading(state) == "generate_comparison"


def test_handle_missing_data_formats_periods_correctly():
    # missing_periods entries carry the same year/quarter keys that
    # decompose_query builds target_periods with (see state.py docstring),
    # not fiscal_year/fiscal_period (those are Document metadata keys, used
    # only inside retrieve_filtered_docs/generate_comparison, not here).
    state = base_state(
        ticker="AAPL",
        missing_periods=[{"year": 2025, "quarter": "Q2"}],
    )
    result = handle_missing_data(state)

    assert "Q2 2025" in result["final_answer"]
    assert result["citations"] == []