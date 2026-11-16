# Benchmarks

## What exists here

`benchmark_retrieval.py` — times `VectorStoreManager.search_filtered`
(query embedding + one metadata-filtered Qdrant query) over N repeated
searches and reports p50 / p95 / mean in milliseconds.

## What it deliberately does NOT measure

End-to-end `/query` latency. A full query also runs query decomposition,
FlashRank reranking, LLM synthesis and grounding verification — budgeted
at **~2.5–4.5 s** end-to-end for a 2-period comparison in
`docs/ARCHITECTURE.md` §4. Any single small millisecond figure quoted for
this project can only ever describe one isolated stage like retrieval,
and must say so.

## How to run

1. Configure a real Qdrant instance (`.env`: `QDRANT_URL`, `QDRANT_API_KEY`).
2. Ingest at least one filing for the ticker/period you will query.
3. Run:

   ```bash
   python benchmarks/benchmark_retrieval.py --queries 50 --ticker AAPL --year 2025 --period Q2
   ```

4. A real run writes `benchmarks/results/retrieval_latency.json`. That
   file is the only legitimate source for a retrieval-latency number.

## Current state

No results file is committed, because no benchmark run against a real,
populated Qdrant instance has been recorded yet. Earlier project
materials quoted latency figures (7.25 ms p50 / 17.76 ms p95) that no
script or results file in this repository ever supported; those figures
have been removed rather than repeated. When a real run exists, quote
its `retrieval_latency.json` — scoped as retrieval-only, with its `n`.

If Qdrant is not configured, the script exits gracefully with an
explanation and writes nothing.
