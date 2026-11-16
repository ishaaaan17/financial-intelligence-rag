"""
Download the latest SEC filing for a ticker straight from EDGAR.

Usage:
    python scripts/fetch_filing.py --ticker AAPL --form 10-Q
    python scripts/fetch_filing.py --ticker MSFT --form 10-K --out data/raw

The file lands in data/raw/ (the parser's staging directory), ready to be
ingested via POST /ingest if it is a PDF, or inspected directly.

Set EDGAR_USER_AGENT to a declared user agent with your contact info —
SEC fair-access policy requires it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edgar import EdgarError, fetch_latest_filing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--form", default="10-Q", help="Filing form (default: 10-Q)")
    parser.add_argument("--out", default="data/raw", help="Destination directory")
    args = parser.parse_args()

    try:
        filing, path = fetch_latest_filing(args.ticker, form=args.form, dest_dir=Path(args.out))
    except EdgarError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{filing.ticker} {filing.form} filed {filing.filing_date} "
          f"(accession {filing.accession_number})")
    print(f"Downloaded: {path}")
    print(f"Source:     {filing.document_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
