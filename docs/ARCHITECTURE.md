# Architecture

System topology, component responsibilities, and latency budgets for the SEC Financial Intelligence RAG Engine.

## 1. Topology

```
┌────────────┐     ┌──────────────┐     ┌───────────────────────────┐
│ Streamlit  │────▶│   FastAPI    │────▶│  LangGraph State Machine  │
│ frontend   │◀────│   backend    │◀────│  (graph.py / nodes.py)    │
└────────────┘     └──────────────┘     └──────────┬────────────────┘
                                                      │        │
                                                      ▼        ▼
                                         ┌────────────────┐  ┌──────────────────┐
                                         │   pdfplumber    │  │   Qdrant Cloud    │
                                         │   ingestion      │─▶│  (vector store)   │
                                         │   pipeline        │  └──────────────────┘
                                         └────────────────┘             │
                                                      │                  ▼
                                                      │        ┌──────────────────┐
                                                      │        │    FlashRank      │
                                                      │        │  cross-encoder    │
                                                      │        └──────────────────┘
                                                      ▼
                                         ┌────────────────┐
                                         │  HuggingFace    │
                                         │  embeddings     │
                                         │  (local, $0)    │
                                         └────────────────┘

             Inference: Groq API (llama-3.3-70b-versatile, temp=0.0)
```

## 2. Component responsibilities

| Component     | File(s)           | Responsibility                                                                                                             |
| ------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| API layer     | `src/app.py`      | FastAPI routes (`/health`, `/ingest`, `/query`), CORS, request/response schemas, dashboard UI                              |
| Config        | `src/config.py`   | Pydantic Settings, `.env` loading, single source of truth for model/collection names                                       |
| Ingestion     | `src/parser.py`   | PDF text + table extraction via `pdfplumber`, Markdown table serialization, chunking                                       |
| Vector store  | `src/database.py` | Qdrant client wrapper, payload indexing (`ticker`, `fiscal_year`, `fiscal_period`, `type`), `query_points` filtered search |
| Orchestration | `src/graph.py`    | Compiled LangGraph state machine, conditional edges between nodes, bounded hallucination-retry loop                        |
| Graph nodes   | `src/nodes.py`    | Query decomposition, retrieval, reranking, relevance grading, synthesis, grounding verification, retry/backoff + logging   |
| State schema  | `src/state.py`    | TypedDict defining the shape of state passed between graph nodes                                                           |
| Frontend      | `frontend.py`     | Streamlit console for ingestion, querying, and citation review                                                             |

## 3. Data flow (per query)

1. **Decompose** — `decompose_query` node extracts `ticker`, `fiscal_year(s)`, `fiscal_period(s)`, and `metric_focus` from the natural-language query using the Groq LLM with structured JSON output.
2. **Retrieve** — for each target period, `database.py` issues a Qdrant `query_points` call filtered on `ticker` + `fiscal_year` + `fiscal_period`, so no chunk from an unrelated quarter can leak into context.
3. **Rerank** — retrieved chunks are reordered locally by the FlashRank cross-encoder (`ms-marco-TinyBERT-L-2-v2`) to push table content above narrative boilerplate.
4. **Grade** — `grade_documents` first checks whether every requested period was actually retrieved (missing periods route to `handle_missing_data` instead of continuing), then runs an LLM relevance grade over the first 3 reranked docs, truncated to 400 chars each — a deliberate cost/latency trade-off, not a full grade of every chunk.
5. **Synthesize** — `generate_comparison` prompts the LLM (temperature 0.0) to produce a Markdown comparison table with absolute and percentage change.
6. **Verify** — `verify_grounding` cross-checks every numerical claim in the generated answer against the raw retrieved context. If ungrounded and a retry attempt remains (`retry_count < MAX_GENERATION_RETRIES`), the graph loops back through `increment_retry` → `generate_comparison` for one more attempt; once the retry budget is exhausted it routes to `handle_hallucination` instead of returning the ungrounded answer.
7. **Respond** — the API returns `answer`, `citations` (ticker/year/period/page/type per chunk used), and `hallucination_detected`.

## 4. Latency budget (target, single query)

| Stage                                    | Target         | Notes                                                                                                                         |
| ---------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Query decomposition                      | ~0.3–0.6s      | Groq structured JSON call                                                                                                     |
| Vector retrieval (per period)            | ~0.1–0.3s      | Qdrant Cloud, indexed payload filters                                                                                         |
| Reranking                                | ~0.05–0.15s    | Local cross-encoder, no network hop                                                                                           |
| Synthesis                                | ~1–2.5s        | Groq, dominant cost for multi-hop (2+ periods)                                                                                |
| Grounding verification                   | ~0.5–1s        | Second LLM pass over generated claims                                                                                         |
| **End-to-end (2-period comparison)**     | **~2.5–4.5s**  | Excludes cold-start / network variance, assumes no retry needed                                                               |
| **+ 1 hallucination retry (worst case)** | **+~1.5–3.5s** | One extra synthesis + verification pass via `increment_retry`, bounded by `MAX_GENERATION_RETRIES` (see `CONSTRAINTS.md` §11) |

Ingestion (`/ingest`) is not on this budget — PDF parsing and embedding of a full 10-Q/10-K typically takes several seconds to ~1 minute depending on document length and table density, and runs outside the query-time path.

## 5. Design rationale

- **Local embeddings + local reranker**: keeps steady-state cost at $0 by avoiding per-query embedding API calls; only synthesis and verification go to Groq.
- **Metadata-isolated retrieval**: filtering by `ticker`/`fiscal_year`/`fiscal_period` at the Qdrant level (not post-filtering) prevents context contamination between quarters, which is the primary source of comparison errors in naive RAG.
- **Two-pass LLM (synthesize → verify)**: trades one extra Groq call for a hard check against hallucinated figures, which is the project's core reliability requirement.
- **Bounded retry + explicit backoff**: a single failed grounding check retries once (`MAX_GENERATION_RETRIES`, DEC-014) before falling back to a safe refusal, and every Groq call goes through a logged retry wrapper with bounded exponential backoff on transient errors only (DEC-015) — see `CONSTRAINTS.md` §11.
