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

## DEC-013: Wildcard CORS Restricted to Local Development

- **Context:** `app.py` currently sets `allow_origins=["*"]` so the Streamlit frontend (and Swagger UI) can call the API freely during development.
- **Decision:** Accept the wildcard for now, but treat it as explicitly temporary — see `CONSTRAINTS.md` §9.
- **Why:** Recorded here specifically so it isn't mistaken for a considered production choice later. Any deployment beyond `localhost` must replace this with an explicit origin allowlist before shipping; this entry exists to make that a tracked decision, not a forgotten default.
