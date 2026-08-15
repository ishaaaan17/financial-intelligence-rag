# Constraints

Non-negotiable technical guardrails for this project. Changes to any of these require an explicit decision recorded in `DECISIONS.md`, not a silent code change.

## 1. Zero-hallucination requirement

- Every numerical claim in a synthesized answer **must** be traceable to a retrieved source chunk via the `verify_grounding` node.
- If grounding verification fails, the graph **must** route to `handle_hallucination` and withhold the ungrounded answer — it must never be returned to the user with a caveat instead of being blocked.
- `hallucination_detected` must always be present and accurate in the `/query` response; it is a contract the frontend depends on to render its verdict banner.

## 2. No cross-period context contamination

- Retrieval **must** be filtered at the vector-database level (`ticker`, `fiscal_year`, `fiscal_period`) — never post-filtered in application code after an unfiltered search.
- A query about Q2 2024 must never receive chunks from Q2 2025 (or any other ticker/period) in its retrieval context, even as "supporting" material.

## 3. Missing data must fail safely

- If a requested `(ticker, fiscal_year, fiscal_period)` combination has no filings ingested, the system **must** return a structured "missing data" response via `handle_missing_data`.
- The system must never fabricate a plausible-sounding figure for a period it has no data for.

## 4. Cost constraints

- The default stack **must** remain $0 at rest: local HuggingFace embeddings, local FlashRank reranking, Qdrant Cloud free tier, Groq free tier.
- Any component that introduces a paid dependency (e.g., swapping in OpenAI embeddings) requires a `DECISIONS.md` entry justifying the cost tradeoff.

## 5. Determinism

- `LLM_TEMPERATURE` is fixed at `0.0` for both synthesis and grounding verification. This is a reliability requirement, not a stylistic default — do not raise it for "more natural" phrasing.

## 6. Table integrity

- Financial tables extracted by `parser.py` **must** preserve column alignment and numerical structure when serialized to Markdown. A parser change that causes numbers to shift columns or merge rows is a regression, not a formatting nit.

## 7. File type & size

- `/ingest` accepts PDF only. Reject non-PDF uploads at the API boundary (`app.py`), not deep in the parsing pipeline.
- Very large filings (multi-hundred-page 10-Ks) must degrade gracefully — chunk and embed incrementally rather than loading the full document into memory at once.

## 8. API contract stability

- `/health`, `/ingest`, `/query` response shapes (`QueryResponse`, `IngestResponse`) are consumed directly by `frontend.py`. Do not change field names or types without updating the frontend in the same change.
- `citations` must always be a list (possibly empty), never `null`, since the frontend builds a DataFrame from it unconditionally when non-empty.

## 9. CORS

- Origins are driven by `settings.allowed_origins` (`ALLOWED_ORIGINS` env var, see `config.py` and `DECISIONS.md` DEC-013) — never hardcode `allow_origins=["*"]` in `app.py` again. The wildcard is acceptable only as a _local_ fallback default value, not as a value ever shipped to a non-local deployment; any non-local deployment must set `ALLOWED_ORIGINS` explicitly to its real frontend origin(s).

## 10. Testing

- Any change to `nodes.py` (query decomposition, grading, synthesis, grounding verification) must be covered by `tests/test_nodes.py` (node-level, mocked LLM/vector store) and, where it affects the `/query` response shape or a graph branch, `tests/test_pipeline.py` (API-level). Both must pass in CI (`.github/workflows/ci.yml`) before merge — these nodes are where hallucination and contamination bugs actually originate.
- Any change to the routing/retry logic in `graph.py` (`route_after_grading`, `route_after_verification`, `increment_retry`, `handle_missing_data`, `handle_hallucination`) must have a corresponding test in `tests/test_nodes.py`, since these functions are pure (state in, route/dict out) and cheap to test directly without mocking the LLM at all.

## 11. Bounded retries — no unbounded loops, no silent retries

- `MAX_GENERATION_RETRIES` in `graph.py` bounds the hallucination-recovery loop (`verify_grounding` → `increment_retry` → `generate_comparison`). Any change to this limit is a `DECISIONS.md`-worthy decision (see DEC-014), not a casual tweak — an unbounded or too-generous retry loop both risks infinite execution and burns Groq calls on a query that isn't going to ground out.
- Groq calls **must** go through `nodes.py`'s `_invoke_chain` wrapper, not a raw `chain.invoke(...)`. Retries there are bounded by `settings.llm_max_retries` and restricted to transient errors (`APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`); the `ChatGroq` client's own `max_retries` must stay `0` so retry behavior isn't split between two invisible layers (see DEC-015).
