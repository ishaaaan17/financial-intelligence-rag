"""
Retrieval-latency benchmark for the Qdrant filtered search.

Measures ONLY `VectorStoreManager.search_filtered` (embedding the query +
one filtered Qdrant query). It does NOT measure end-to-end /query latency,
which additionally includes query decomposition, reranking, synthesis and
grounding verification (budgeted at ~2.5-4.5s in docs/ARCHITECTURE.md §4).

Requirements: QDRANT_URL and QDRANT_API_KEY must be configured (via .env),
and the collection must already contain filings to search. If Qdrant is
not configured, this script prints what is missing and exits 0 without
writing any results — it never fabricates numbers.

Usage:
    python benchmarks/benchmark_retrieval.py [--queries N] [--ticker AAPL]

Results (only from a real run) are written to
benchmarks/results/retrieval_latency.json.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_QUERIES = [
    "Research and development spend",
    "Total net sales",
    "Gross margin",
    "Operating income",
    "Cost of sales",
    "Selling general and administrative expenses",
    "Net income",
    "Earnings per share diluted",
]


def percentile(values, pct):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=50, help="Number of timed searches")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--period", default="Q2")
    args = parser.parse_args()

    from src.config import settings

    if not settings.qdrant_url or not settings.qdrant_api_key:
        print("Qdrant is not configured (QDRANT_URL / QDRANT_API_KEY missing).")
        print("Nothing was measured and no results file was written.")
        print("Configure a real Qdrant instance with ingested filings, then re-run.")
        return 0

    from src.database import VectorStoreManager

    store = VectorStoreManager()
    latencies_ms = []
    for i in range(args.queries):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        start = time.perf_counter()
        store.search_filtered(
            query=query, ticker=args.ticker,
            fiscal_year=args.year, fiscal_period=args.period, limit=5,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000)

    results = {
        "what_was_measured": "VectorStoreManager.search_filtered only "
                             "(query embedding + one filtered Qdrant query_points call). "
                             "NOT end-to-end /query latency.",
        "n": len(latencies_ms),
        "ticker": args.ticker,
        "fiscal_year": args.year,
        "fiscal_period": args.period,
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
    }

    out = Path(__file__).parent / "results" / "retrieval_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
