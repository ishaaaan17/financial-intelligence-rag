"""
LangGraph State schema definition.
Defines the shared state dictionary passed across all workflow nodes.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Shared state object representing the full context of a financial query session.
    """
    # Original user input
    original_query: str

    # Extracted structured query plan
    ticker: str
    target_periods: List[Dict[str, Any]]  # e.g., [{"year": 2024, "quarter": "Q2"}, {"year": 2025, "quarter": "Q2"}]
    metric_focus: str                    # e.g., "R&D expense", "Operating margin"

    # Retrieved data containers
    retrieved_docs: List[Document]
    missing_periods: List[Dict[str, Any]]

    # Quality & self-correction tracking
    documents_are_relevant: bool
    hallucination_detected: bool
    retry_count: int

    # Final outputs
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]