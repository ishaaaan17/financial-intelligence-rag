# Architectural & Technology Decisions

## DEC-001: Pydantic Settings for Configuration Management

- **Context:** Need a safe way to load API keys and paths across multiple modules.
- **Decision:** Use `pydantic-settings` to enforce schema validation at startup.
- **Why:** Halts application startup immediately if required keys (like `GROQ_API_KEY`) are missing, preventing late runtime failures.

## DEC-002: Deterministic Inference (Temperature = 0.0)

- **Context:** Financial SEC extraction requires exact numerical precision.
- **Decision:** Lock LLM temperature to `0.0`.
- **Why:** Eliminates stylistic randomness and reduces hallucination risk when parsing balance sheets and income statements.

## DEC-003: Groq + Local HuggingFace Embeddings ($0 Stack)

- **Context:** Keep production deployment and local development completely free.
- **Decision:** Use Groq Cloud free tier (`llama-3.3-70b-versatile`) for reasoning and `sentence-transformers/all-MiniLM-L6-v2` locally for embeddings.
- **Why:** High-speed inference without credit card requirements or vendor lock-in.

## DEC-004: TypedDict Schema for Graph State

- **Context:** State must be passed deterministically across multiple agentic nodes.
- **Decision:** Use a strictly typed `AgentState(TypedDict)` containing query decompose targets, document collections, retry counters, and hallucination flags.
- **Why:** Prevents state mutation bugs, provides IDE autocompletion, and bounds retry loops to prevent infinite execution cycles.

## DEC-005: Metadata-Isolated Retrieval at the Database Level

- **Context:** Multi-hop comparative queries (e.g. Q2 2024 vs Q2 2025) retrieve from the same collection for both periods. A naive unfiltered similarity search can pull chunks from the wrong quarter or wrong company if their embeddings happen to be close.
- **Decision:** Enforce `ticker`, `fiscal_year`, and `fiscal_period` as indexed Qdrant payload filters applied _at query time_ (`query_points` with a `Filter`), never as a post-hoc filter over an unfiltered result set in application code.
- **Why:** Filtering at the database level is both correctness-critical and cheaper — it prevents cross-period contamination by construction rather than by hoping the top-k similarity results happen to be clean, and it avoids retrieving (and paying reranking cost on) irrelevant chunks in the first place. This is the primary defense against the project's main failure mode: confidently comparing the wrong quarter's numbers.

## DEC-006: FlashRank Local Cross-Encoder Reranking

- **Context:** SEC filings are dominated by narrative disclosure (risk factors, legal boilerplate) relative to the actual financial tables a query cares about. Vector similarity alone often ranks boilerplate above the relevant table.
- **Decision:** Add a local reranking pass (`ms-marco-TinyBERT-L-2-v2` via FlashRank) between retrieval and synthesis, reordering chunks by relevance to the decomposed query before they reach the LLM.
- **Why:** Runs locally in milliseconds with no API cost or network hop, and materially improves the odds that the actual R&D/revenue/expense table — not surrounding prose — ends up in the synthesis context. Rejected alternative: skipping reranking and relying on a larger top-k, which increases token cost and dilutes the synthesis context with noise.

## DEC-007: Two-Pass LLM — Synthesize, Then Independently Verify

- **Context:** A single LLM call that both generates and "double-checks" its own answer is a weak guardrail — self-verification within the same generation is prone to confirming its own mistakes.
- **Decision:** Split synthesis and verification into two separate graph nodes (`generate_comparison` → `verify_grounding`) with the verification pass explicitly cross-referencing generated numerical claims against raw retrieved chunk text, independent of the generation reasoning.
- **Why:** A dedicated verification pass with its own prompt and context framing catches ungrounded figures that a single combined generate-and-check call would tend to wave through. The extra Groq call is accepted as a deliberate reliability cost, not an oversight (see `CONSTRAINTS.md` §1).

## DEC-008: LangGraph for Orchestration

- **Context:** The query pipeline is not linear — it needs conditional branching (missing data vs. found data, grounded vs. hallucinated) and bounded retry loops, not just a fixed chain of LLM calls.
- **Decision:** Use LangGraph's `StateGraph` with explicit conditional edges rather than a hand-rolled sequence of function calls or a simpler linear chain (e.g. plain LangChain `LCEL`).
- **Why:** Conditional routing (`grade_documents`, `verify_grounding`) and retry-count-bounded loops are first-class in LangGraph's edge model, making the safe-failure paths (`handle_missing_data`, `handle_hallucination`) explicit and testable as graph transitions rather than nested `if/else` scattered through application code.

## DEC-009: pdfplumber for SEC Filing Parsing

- **Context:** SEC 10-Q/10-K PDFs mix narrative text with dense financial tables where column alignment carries meaning (a number in the wrong column is a wrong number).
- **Decision:** Use `pdfplumber` for extraction, with explicit table-detection and Markdown table serialization, rather than a generic text-only PDF loader.
- **Why:** Generic text extraction collapses tables into unstructured text and loses column alignment, which is unacceptable for financial data (see `CONSTRAINTS.md` §6). `pdfplumber`'s table-detection lets the parser preserve rows/columns and tag chunks by `type` (table vs. narrative), which downstream reranking depends on to prioritize tables.

## DEC-010: Fail-Safe Handling of Missing Filing Data

- **Context:** If a user asks about a period that hasn't been ingested, the system must not guess.
- **Decision:** `grade_documents` explicitly checks for missing target periods and routes to a dedicated `handle_missing_data` node that returns a structured "not available" response, rather than letting an empty or thin retrieval result flow into synthesis.
- **Why:** An LLM given a weak or empty context for "Q2 2023 R&D spend" will often still produce a plausible-sounding number from parametric memory rather than admitting it has nothing. Making the missing-data path a distinct, tested graph branch (rather than relying on the synthesis prompt to "be honest") removes that failure mode structurally.

## DEC-011: Qdrant Cloud as the Vector Store

- **Context:** Needed a vector database supporting rich payload filtering (not just similarity search) and a genuinely free tier for a $0 stack.
- **Decision:** Use Qdrant Cloud with explicit payload indexing on `ticker` (Keyword), `fiscal_year` (Integer), `fiscal_period` (Keyword), and `type` (Keyword).
- **Why:** Qdrant's filtered search is a first-class query primitive (not a bolt-on), which DEC-005's metadata isolation depends on. Payload indexing on the filtered fields keeps filtered queries fast as the collection grows, rather than degrading to a linear scan.

## DEC-012: Streamlit for the Frontend

- **Context:** Needed an internal-facing console for uploading filings and running queries, without building a separate JS frontend and API client.
- **Decision:** Use Streamlit (`frontend.py`) talking to the FastAPI backend over plain HTTP (`httpx`), rather than a React/Next.js frontend.
- **Why:** Streamlit lets the ingestion form, query box, and citation table be built and iterated in pure Python, matching the project's Python-only stack and small-team/internal-tool scope. Explicitly revisit this if the tool needs to be exposed to external, non-technical users at scale — Streamlit's session model and auth story are not built for that.

## DEC-013: Wildcard CORS Restricted to Local Development [RESOLVED]

- **Context:** `app.py` originally set `allow_origins=["*"]` so the Streamlit frontend (and Swagger UI) could call the API freely during development.
- **Decision:** Replaced the wildcard with `settings.allowed_origins`, a parsed allowlist driven by the `ALLOWED_ORIGINS` env var (see `config.py`), defaulting to the local Streamlit + Swagger dev origins.
- **Why:** Recorded here specifically so the wildcard isn't mistaken for a considered production choice. This entry is now resolved in code — see `CONSTRAINTS.md` §9 for the deployment guardrail this satisfies. Any non-local deployment still needs its actual frontend origin(s) set via `ALLOWED_ORIGINS`.

## DEC-014: Bounded Retry Loop for Hallucination Recovery

- **Context:** `AgentState` has always carried a `retry_count` field, and earlier docs (`FLOW.md`, `HANDOVER.md`) described `handle_hallucination` as retrying `generate_comparison` before giving up — but the graph never actually implemented this; `handle_hallucination` returned a static fallback and routed straight to `END`, and `retry_count` was never read or incremented.
- **Decision:** Implement the real bounded retry: `route_after_verification` checks `retry_count` against `MAX_GENERATION_RETRIES` (currently `1`, i.e. 2 total generation attempts). If budget remains, route to a new `increment_retry` node that bumps the counter and loops back to `generate_comparison`; once exhausted, route to `handle_hallucination`, which now reports the actual number of attempts made.
- **Why:** A single failed grounding check can be a one-off (borderline phrasing, a marginal figure) rather than a structural problem with the retrieved context — retrying once cheaply recovers those cases instead of immediately surfacing a refusal. The retry is still hard-bounded to avoid the infinite-loop risk `DEC-004` was already designed to prevent; `MAX_GENERATION_RETRIES` is a small, explicit module-level constant in `graph.py`, not a magic number buried in the routing function.

## DEC-015: Explicit Bounded Backoff for Groq Calls, Structured Per-Node Logging

- **Context:** `nodes.py` called `chain.invoke(...)` directly with no timeout and no retry handling. A transient Groq error (connection blip, rate limit, 5xx) threw an unhandled exception straight up through the graph, and there was no logging anywhere in `nodes.py`/`graph.py` to diagnose a failure or a hallucination-triggered retry after the fact.
- **Decision:** Add `timeout=settings.llm_timeout_seconds` to the `ChatGroq` client and set its own `max_retries=0`, so retry behavior is handled entirely by an explicit wrapper (`_invoke_chain` in `nodes.py`) using `tenacity` with exponential backoff, bounded by `settings.llm_max_retries`, and retrying only genuinely transient errors (`APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`) — not auth/validation/bad-request errors, which no amount of retrying fixes. Every node is also wrapped with a `log_node` decorator emitting structured start/end/error logs (node name, ticker, retry_count, duration, status).
- **Why:** Disabling the client's own silent retries and routing everything through one logged wrapper means every retry attempt is visible in logs rather than happening invisibly inside the Groq client. Restricting retries to transient error types avoids wasting the retry budget (and adding latency) on errors a retry can't fix. Per-node logging gives operators a way to see which node failed, how long it took, and how many hallucination-retry cycles a given query went through — none of which was observable before.
