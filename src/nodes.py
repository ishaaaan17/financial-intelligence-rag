"""
LangGraph execution nodes for the Financial RAG pipeline.
Includes query planning, filtered retrieval, FlashRank reranking, relevance grading, comparison synthesis, and grounding verification.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from src.config import settings
from src.state import AgentState
from src.database import VectorStoreManager

# Initialize LLM and Vector Store
llm = ChatGroq(
    model=settings.llm_model,
    api_key=settings.groq_api_key,
    temperature=settings.llm_temperature
)
vector_store = VectorStoreManager()
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")


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
def decompose_query(state: AgentState) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(QueryPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract structured query parameters from the user prompt."),
        ("human", "{query}")
    ])
    chain = prompt | structured_llm
    result: QueryPlan = chain.invoke({"query": state["original_query"]})

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
    grade: RelevanceGrade = chain.invoke({"metric": state["metric_focus"], "context": sample_context})

    return {"documents_are_relevant": grade.is_relevant}


# Node 4: Generate Multi-Hop Comparison Table
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
    response = chain.invoke({"query": state["original_query"], "context": context_str})

    return {
        "final_answer": response.content,
        "citations": citations
    }


# Node 5: Verify Grounding (Anti-Hallucination)
def verify_grounding(state: AgentState) -> Dict[str, Any]:
    structured_llm = llm.with_structured_output(GroundingVerification)
    context_str = "\n".join([d.page_content for d in state["retrieved_docs"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify that all facts and numbers in the answer are strictly supported by the raw context."),
        ("human", "Raw Context:\n{context}\n\nGenerated Answer:\n{answer}")
    ])

    chain = prompt | structured_llm
    verification: GroundingVerification = chain.invoke({
        "context": context_str,
        "answer": state.get("final_answer", "")
    })

    return {
        "hallucination_detected": not verification.is_grounded
    }