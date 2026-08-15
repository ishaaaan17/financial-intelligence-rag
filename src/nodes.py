"""
LangGraph execution nodes for the Financial RAG pipeline.
Includes query planning, filtered retrieval, FlashRank reranking, relevance grading, comparison synthesis, and grounding verification.
"""

import functools
import logging
import time
from typing import List, Dict, Any, Callable

from groq import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config import settings
from src.state import AgentState
from src.database import VectorStoreManager

logger = logging.getLogger("financial_rag.nodes")

# Initialize LLM and Vector Store
# max_retries=0: the langchain-groq/groq client's built-in retry is disabled here
# so that all retry/backoff behavior goes through the explicit, logged wrapper below
# (_invoke_chain) instead of retrying silently with no visibility.
llm = ChatGroq(
    model=settings.llm_model,
    api_key=settings.groq_api_key,
    temperature=settings.llm_temperature,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
)
vector_store = VectorStoreManager()
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")


# Retryable Groq call wrapper
# Only transient/network-ish failures are retried. Non-transient errors (bad request,
# auth, permission, not found, validation) are intentionally NOT retried since a retry
# cannot fix them and would just waste the retry budget and add latency.
_RETRYABLE_GROQ_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def _build_groq_retry():
    return retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE_GROQ_ERRORS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _invoke_chain(chain, payload: Dict[str, Any]):
    """
    Invokes an LCEL chain (prompt | llm) with bounded retries and exponential
    backoff on transient Groq errors (connection issues, timeouts, rate limits,
    5xx). Non-transient errors propagate immediately without retrying.
    """
    groq_retry = _build_groq_retry()
    return groq_retry(chain.invoke)(payload)


# Structured per-node logging decorator
def log_node(node_name: str) -> Callable:
    """
    Wraps a graph node function with structured start/end/error logging.
    Logs node name, ticker, and retry_count on entry; duration and status on exit.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> Dict[str, Any]:
            start = time.monotonic()
            logger.info(
                "node_start",
                extra={
                    "node": node_name,
                    "ticker": state.get("ticker"),
                    "retry_count": state.get("retry_count", 0),
                },
            )
            try:
                result = fn(state)
                logger.info(
                    "node_end",
                    extra={
                        "node": node_name,
                        "status": "ok",
                        "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    },
                )
                return result
            except Exception:
                logger.exception(
                    "node_error",
                    extra={
                        "node": node_name,
                        "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    },
                )
                raise
        return wrapper
    return decorator


# Pydantic Schemas for Structured LLM Outputs
class QueryPlan(BaseModel):
    ticker: str = Field(description="The primary stock ticker symbol in uppercase, e.g., 'AAPL', 'MSFT'")
    years: List[int] = Field(description="Target fiscal years mentioned in the query, e.g., [2024, 2025]")
    quarters: List[str] = Field(description="Target fiscal quarters mentioned, e.g., ['Q2', 'Q2']")
    metric_focus: str = Field(description="The core financial metric to compare, e.g., 'Research and development expense'")


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if the text contains financial figures relevant to the query metric.")


class GroundingVerification(BaseModel):
    is_grounded: bool = Field(description="True if every number/statistic in the answer exists in the source text.")
    explanation: str = Field(description="Brief reason for verification decision.")


# Node 1: Decompose User Query
@log_node("decompose_query")
def decompose_query(state: AgentState) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(QueryPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract structured query parameters from the user prompt."),
        ("human", "{query}")
    ])
    chain = prompt | structured_llm
    result: QueryPlan = _invoke_chain(chain, {"query": state["original_query"]})

    target_periods = []
    if len(result.years) == len(result.quarters):
        for y, q in zip(result.years, result.quarters):
            p = {"year": int(y), "quarter": q.upper()}
            if p not in target_periods:
                target_periods.append(p)
    else:
        for y in result.years:
            for q in result.quarters:
                p = {"year": int(y), "quarter": q.upper()}
                if p not in target_periods:
                    target_periods.append(p)

    return {
        "ticker": result.ticker.upper(),
        "metric_focus": result.metric_focus,
        "target_periods": target_periods,
        "retry_count": state.get("retry_count", 0),
        "retrieved_docs": []
    }


# Node 2: Retrieve Documents & Apply FlashRank Reranking
@log_node("retrieve_filtered_docs")
def retrieve_filtered_docs(state: AgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    metric = state["metric_focus"]
    all_docs: List[Document] = []
    missing = []

    for period in state["target_periods"]:
        year = period["year"]
        quarter = period["quarter"]

        docs = vector_store.search_filtered(
            query=metric,
            ticker=ticker,
            fiscal_year=year,
            fiscal_period=quarter,
            limit=5
        )

        if not docs:
            missing.append(period)
        else:
            # Re-rank retrieved passages using FlashRank Cross-Encoder
            passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
            rerank_request = RerankRequest(query=metric, passages=passages)
            ranked_passages = ranker.rerank(rerank_request)

            top_docs = [
                Document(page_content=p["text"], metadata=p["meta"])
                for p in ranked_passages[:3]
            ]
            all_docs.extend(top_docs)

    return {
        "retrieved_docs": all_docs,
        "missing_periods": missing
    }


# Node 3: Grade Relevance
@log_node("grade_documents")
def grade_documents(state: AgentState) -> Dict[str, Any]:
    if state["missing_periods"] or not state["retrieved_docs"]:
        return {"documents_are_relevant": False}

    structured_llm = llm.with_structured_output(RelevanceGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade if the provided context contains data to answer the metric query: '{metric}'."),
        ("human", "Context:\n{context}")
    ])
    chain = prompt | structured_llm

    sample_context = "\n---\n".join([d.page_content[:400] for d in state["retrieved_docs"][:3]])
    grade: RelevanceGrade = _invoke_chain(chain, {"metric": state["metric_focus"], "context": sample_context})

    return {"documents_are_relevant": grade.is_relevant}


# Node 4: Generate Multi-Hop Comparison Table
@log_node("generate_comparison")
def generate_comparison(state: AgentState) -> Dict[str, Any]:
    context_chunks = []
    citations = []

    for d in state["retrieved_docs"]:
        context_chunks.append(f"[{d.metadata.get('fiscal_period')} {d.metadata.get('fiscal_year')}]:\n{d.page_content}")
        meta = {
            "ticker": d.metadata.get("ticker"),
            "fiscal_year": d.metadata.get("fiscal_year"),
            "fiscal_period": d.metadata.get("fiscal_period"),
            "page": d.metadata.get("page"),
            "type": d.metadata.get("type")
        }
        if meta not in citations:
            citations.append(meta)

    context_str = "\n\n".join(context_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict SEC financial analyst. Build a clean markdown table comparing the metric across "
            "the requested periods. Include calculated YoY absolute change and percentage change. "
            "ONLY use exact figures present in the context. Never fabricate data."
        )),
        ("human", "User Query: {query}\n\nRetrieved SEC Context:\n{context}")
    ])

    chain = prompt | llm
    response = _invoke_chain(chain, {"query": state["original_query"], "context": context_str})

    return {
        "final_answer": response.content,
        "citations": citations
    }


# Node 5: Verify Grounding (Anti-Hallucination)
@log_node("verify_grounding")
def verify_grounding(state: AgentState) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(GroundingVerification)
    context_str = "\n".join([d.page_content for d in state["retrieved_docs"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify that all facts and numbers in the answer are strictly supported by the raw context."),
        ("human", "Raw Context:\n{context}\n\nGenerated Answer:\n{answer}")
    ])

    chain = prompt | structured_llm
    verification: GroundingVerification = _invoke_chain(chain, {
        "context": context_str,
        "answer": state.get("final_answer", "")
    })

    return {
        "hallucination_detected": not verification.is_grounded
    }