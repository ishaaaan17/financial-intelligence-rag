"""
Node implementations for LangGraph Financial RAG.
Contains pure functions that read from AgentState and return state updates.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings
from src.state import AgentState
from src.database import VectorStoreManager

# 1. Initialize LLM & Vector Store
llm = ChatGroq(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    groq_api_key=settings.groq_api_key
)
vector_store = VectorStoreManager()


# 2. Pydantic Schemas for Structured LLM Outputs
class QueryPlan(BaseModel):
    ticker: str = Field(description="Stock ticker symbol in uppercase, e.g. AAPL")
    metric_focus: str = Field(description="The primary financial metric requested, e.g. R&D spend, Gross Margin")
    years: List[int] = Field(description="List of fiscal years mentioned, e.g. [2024, 2025]")
    quarters: List[str] = Field(description="List of quarters mentioned in uppercase, e.g. ['Q2', 'Q2'] or ['FY']")


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if context contains numbers relevant to the metric, False otherwise")
    feedback: str = Field(description="Brief reason for grading decision")


class VerificationGrade(BaseModel):
    is_grounded: bool = Field(description="True if all numbers in generation exist in source context, False if hallucinated")


# 3. Node Functions

def decompose_query(state: AgentState) -> Dict[str, Any]:
    """Extracts ticker, periods, and metrics from the raw user query."""
    structured_llm = llm.with_structured_output(QueryPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract structured query parameters from the user prompt."),
        ("human", "{query}")
    ])

    chain = prompt | structured_llm
    result: QueryPlan = chain.invoke({"query": state["original_query"]})

    # Assemble structured periods
    target_periods = []
    for y in result.years:
        for q in result.quarters:
            target_periods.append({"year": y, "quarter": q.upper()})

    return {
        "ticker": result.ticker.upper(),
        "metric_focus": result.metric_focus,
        "target_periods": target_periods,
        "retry_count": state.get("retry_count", 0),
        "retrieved_docs": []
    }


def retrieve_filtered_docs(state: AgentState) -> Dict[str, Any]:
    """Retrieves document chunks using exact metadata filters for each target period."""
    all_docs = []
    missing_periods = []

    for period in state["target_periods"]:
        docs = vector_store.search_filtered(
            query=state["metric_focus"],
            ticker=state["ticker"],
            fiscal_year=period["year"],
            fiscal_period=period["quarter"],
            limit=3
        )
        if docs:
            all_docs.extend(docs)
        else:
            missing_periods.append(period)

    return {
        "retrieved_docs": all_docs,
        "missing_periods": missing_periods
    }


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Evaluates whether retrieved documents provide enough data to answer the query."""
    if not state["retrieved_docs"]:
        return {"documents_are_relevant": False}

    combined_context = "\n\n".join([d.page_content for d in state["retrieved_docs"]])

    structured_llm = llm.with_structured_output(RelevanceGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Check if the context contains numerical or factual data regarding '{metric}' for {ticker}."),
        ("human", "Context:\n{context}\n\nMetric: {metric}")
    ])

    chain = prompt | structured_llm
    result: RelevanceGrade = chain.invoke({
        "ticker": state["ticker"],
        "metric": state["metric_focus"],
        "context": combined_context
    })

    return {"documents_are_relevant": result.is_relevant}


def generate_comparison(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the financial comparison based strictly on retrieved documents."""
    combined_context = "\n\n".join([
        f"[{d.metadata.get('fiscal_period')} {d.metadata.get('fiscal_year')}]:\n{d.page_content}"
        for d in state["retrieved_docs"]
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict quantitative financial analyst. Compare the requested metrics across periods.\n"
            "Rules:\n"
            "1. Output a clear Markdown comparison table.\n"
            "2. Calculate absolute change ($) and percentage change (%).\n"
            "3. Cite exact sources and footnotes.\n"
            "4. NEVER guess or extrapolate numbers not present in context."
        )),
        ("human", "User Question: {query}\n\nRetrieved Filings Context:\n{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": state["original_query"],
        "context": combined_context
    })

    citations = [d.metadata for d in state["retrieved_docs"]]

    return {
        "final_answer": response.content,
        "citations": citations
    }


def verify_grounding(state: AgentState) -> Dict[str, Any]:
    """Checks if the generated output has any hallucinations against source context."""
    if not state.get("final_answer"):
        return {"hallucination_detected": True}

    combined_context = "\n\n".join([d.page_content for d in state["retrieved_docs"]])

    structured_llm = llm.with_structured_output(VerificationGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify that EVERY numerical claim in the answer is directly supported by the context. Score is_grounded as False if any number is made up."),
        ("human", "Context:\n{context}\n\nAnswer to Verify:\n{answer}")
    ])

    chain = prompt | structured_llm
    result: VerificationGrade = chain.invoke({
        "context": combined_context,
        "answer": state["final_answer"]
    })

    return {"hallucination_detected": not result.is_grounded}