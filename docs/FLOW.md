# Query Lifecycle

Step-by-step walkthrough of a single request through the system, for both the ingestion path and the query path.

## A. Ingestion flow (`POST /ingest`)

1. **Upload** — client (Streamlit or Swagger) sends `multipart/form-data`: `ticker`, `fiscal_year`, `fiscal_period`, `file` (PDF).
2. **Validate extension** — `app.py` rejects anything not ending in `.pdf` with `400`.
3. **Stage to disk** — the upload is streamed to a `NamedTemporaryFile` so `pdfplumber` can read it by path.
4. **Parse** — `SECParser(ticker, fiscal_year, fiscal_period).parse_pdf(tmp_path)`:
   - Walks each page of the PDF.
   - Extracts narrative text and detected tables separately.
   - Serializes tables into Markdown, preserving column order.
   - Produces a list of tagged chunks (each carrying `ticker`/`fiscal_year`/`fiscal_period`/`type`/`page`).
5. **Guard against empty extraction** — if no chunks were produced (e.g., a scanned/image-only PDF), return `400` rather than silently storing nothing.
6. **Embed + upsert** — `vector_store.insert_chunks(chunks)` embeds each chunk locally (`all-MiniLM-L6-v2`) and upserts into the Qdrant collection with payload indexing on `ticker`, `fiscal_year`, `fiscal_period`, `type`.
7. **Cleanup** — the temp file is deleted in a `finally` block regardless of success or failure.
8. **Respond** — `IngestResponse` with `chunks_stored` count.

## B. Query flow (`POST /query`)

1. **Request in** — client sends `{"query": "..."}`. `app.py` builds the initial `GraphState` (all fields empty/false/zero) and invokes `financial_rag_app`.

2. **`decompose_query` node**
   - Sends the raw query to Groq (via `_invoke_chain`, which applies a bounded timeout and retry-with-backoff on transient errors — see `CONSTRAINTS.md` §11) with a structured JSON schema.
   - Extracts `ticker`, one or more `(fiscal_year, fiscal_period)` targets, and `metric_focus`.
   - Example: _"Compare Apple's R&D spend between Q2 2024 and Q2 2025"_ → `ticker=AAPL`, `target_periods=[(2024,Q2),(2025,Q2)]`, `metric_focus="R&D spend"`.

3. **Retrieval + reranking** (`retrieve_filtered_docs` node — one node does both, in this order)
   - For each target period, query Qdrant with a payload filter on `ticker` + `fiscal_year` + `fiscal_period` (top 5 per period).
   - If the collection is empty for a given filter combination, that period is added to `missing_periods` instead of erroring.
   - FlashRank cross-encoder then reranks that period's retrieved chunks against `metric_focus`, and only the **top 3** per period continue. Reranking happens here, inside retrieval — *before* grading, not after it.

4. **`grade_documents` node** (conditional edge)
   - If `missing_periods` is non-empty → route to **`handle_missing_data`** → structured "we don't have this filing" response → `END`.
   - Otherwise an LLM relevance grade runs over the first 3 reranked docs (truncated to 400 chars each — a deliberate cost/latency trade-off). If they don't actually address `metric_focus` → same missing/irrelevant path.
   - Otherwise → continue to synthesis.

5. **`generate_comparison` node**
   - Groq (temperature 0.0) synthesizes a Markdown answer: a comparison table with absolute and percentage change, plus brief prose framing.
   - Citations are attached per fact, referencing the source chunk's `ticker`/`fiscal_year`/`fiscal_period`/`page`/`type`.

6. **`verify_grounding` node** (conditional edge)
   - Every numeric claim in the generated answer is checked against the raw retrieved chunk text.
   - If all claims are grounded → `hallucination_detected=False` → `END`.
   - If any claim can't be matched → `hallucination_detected=True` → `route_after_verification` checks `retry_count` against `MAX_GENERATION_RETRIES` (currently `1`, i.e. 2 total generation attempts):
     - **Retry budget remaining** → route to **`increment_retry`**, which bumps `retry_count` and loops back to `generate_comparison` for another attempt (same prompt, freshly regenerated — not a modified "stricter" prompt).
     - **Retry budget exhausted** → route to **`handle_hallucination`**, which returns a safe fallback response (reporting how many attempts were made) rather than the ungrounded answer, and the graph ends.

7. **Response out**
   - `app.py` reads `final_output` from the completed graph state and returns `QueryResponse`: `query`, `ticker`, `answer`, `citations`, `hallucination_detected`.

## C. Frontend consumption

- **Sidebar**: polls `/health` on page load to show backend status, active model, and collection name; renders the ingestion form and calls `/ingest`.
- **Main panel**: takes the query text, calls `/query`, then:
  - Renders a verdict banner from `hallucination_detected`.
  - Renders `answer` as Markdown (so the comparison table displays natively).
  - Renders `citations` as a dataframe inside a collapsed expander for provenance review.
