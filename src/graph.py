"""
LangGraph workflow definition.
Assembles state nodes and conditional edges into a compiled executable agent.
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    decompose_query,
    retrieve_filtered_docs,
    grade_documents,
    generate_comparison,
    verify_grounding
)

# Maximum number of retries allowed after the first generate_comparison attempt.
# 1 retry => 2 total attempts at generation before falling back to handle_hallucination.
MAX_GENERATION_RETRIES = 1


# 1. Routing and Decision Functions

def route_after_grading(state: AgentState) -> str:
    """
    Decides whether to proceed to generation or halt due to missing/irrelevant context.
    """
    if not state.get("documents_are_relevant", False):
        return "handle_missing_data"
    return "generate_comparison"


def route_after_verification(state: AgentState) -> str:
    """
    Decides whether the final answer is verified, should be retried, or has
    exhausted its retry budget and must fall back to a safe refusal.
    """
    if state.get("hallucination_detected", False):
        if state.get("retry_count", 0) < MAX_GENERATION_RETRIES:
            return "increment_retry"
        return "handle_hallucination"
    return END


# 2. Fallback / Handler Nodes

def handle_missing_data(state: AgentState) -> dict:
    """Fallback node when required filing quarters are unavailable in the vector store."""
    missing = state.get("missing_periods", [])
    periods_str = ", ".join([f"{p.get('quarter')} {p.get('year')}" for p in missing])

    return {
        "final_answer": (
            f"Insufficient data in vector store for {state.get('ticker', 'requested company')}. "
            f"Could not find verified filing chunks for periods: {periods_str or 'requested quarters'}. "
            "Please ensure the corresponding SEC 10-Q/10-K PDFs have been ingested."
        ),
        "citations": []
    }


def increment_retry(state: AgentState) -> dict:
    """Bumps retry_count before looping back to generate_comparison for another attempt."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def handle_hallucination(state: AgentState) -> dict:
    """Fallback node when generated answer failed factual grounding checks after all retries."""
    attempts = state.get("retry_count", 0) + 1
    return {
        "final_answer": (
            f"Safety Warning: The generated comparative metrics could not be verified "
            f"against the raw SEC source tables after {attempts} attempt(s). "
            "Generation blocked to maintain zero-hallucination policy."
        )
    }


# 3. Build the StateGraph

def create_financial_rag_graph():
    workflow = StateGraph(AgentState)

    # Add core nodes
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve_filtered_docs", retrieve_filtered_docs)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_comparison", generate_comparison)
    workflow.add_node("verify_grounding", verify_grounding)

    # Add fallback / retry nodes
    workflow.add_node("handle_missing_data", handle_missing_data)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("handle_hallucination", handle_hallucination)

    # Define execution edges
    workflow.set_entry_point("decompose_query")
    workflow.add_edge("decompose_query", "retrieve_filtered_docs")
    workflow.add_edge("retrieve_filtered_docs", "grade_documents")

    # Conditional branching after document grading
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_comparison": "generate_comparison",
            "handle_missing_data": "handle_missing_data"
        }
    )

    # Edge from generation to grounding verification
    workflow.add_edge("generate_comparison", "verify_grounding")

    # Conditional branching after verification: verified -> END,
    # retry budget remaining -> increment_retry -> generate_comparison again,
    # retry budget exhausted -> handle_hallucination
    workflow.add_conditional_edges(
        "verify_grounding",
        route_after_verification,
        {
            END: END,
            "increment_retry": "increment_retry",
            "handle_hallucination": "handle_hallucination"
        }
    )

    # Retry loop: bump counter, then try generation again
    workflow.add_edge("increment_retry", "generate_comparison")

    # Fallback nodes route to END
    workflow.add_edge("handle_missing_data", END)
    workflow.add_edge("handle_hallucination", END)

    return workflow.compile()


# Compiled runnable graph instance
financial_rag_app = create_financial_rag_graph()