"""
Tests for the SEC EDGAR client (src/edgar.py).

All HTTP is faked with httpx.MockTransport — no live network, no SEC
rate limits, deterministic payloads shaped like the real EDGAR APIs.
"""

import httpx
import pytest

from src.edgar import (
    EdgarError,
    FilingRef,
    fetch_latest_filing,
    find_latest_filing,
    get_cik,
)

TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}

SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "10-Q"],
            "accessionNumber": ["0000320193-26-000001", "0000320193-25-000108", "0000320193-25-000073"],
            "filingDate": ["2026-01-15", "2025-10-31", "2025-08-01"],
            "reportDate": ["2026-01-10", "2025-09-27", "2025-06-28"],
            "primaryDocument": ["aapl-8k.htm", "aapl-20250927.htm", "aapl-20250628.htm"],
        }
    }
}


def _mock_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS_PAYLOAD)
        if "/submissions/CIK" in url:
            return httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
        if "/Archives/edgar/data/" in url:
            return httpx.Response(200, content=b"<html>filing body</html>")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_cik_zero_pads_to_ten_digits():
    with _mock_client() as client:
        assert get_cik("AAPL", client=client) == "0000320193"
        assert get_cik("msft", client=client) == "0000789019"  # case-insensitive


def test_get_cik_unknown_ticker_raises():
    with _mock_client() as client:
        with pytest.raises(EdgarError, match="not found"):
            get_cik("ZZZZ", client=client)


def test_find_latest_filing_skips_other_forms():
    filing = find_latest_filing(SUBMISSIONS_PAYLOAD, form="10-Q", ticker="AAPL", cik="0000320193")
    # The 8-K is newer but is not a 10-Q; the first 10-Q must win.
    assert filing.accession_number == "0000320193-25-000108"
    assert filing.report_date == "2025-09-27"
    assert filing.document_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019325000108/aapl-20250927.htm"
    )


def test_find_latest_filing_missing_form_raises():
    with pytest.raises(EdgarError, match="No 10-K"):
        find_latest_filing(SUBMISSIONS_PAYLOAD, form="10-K")


def test_fetch_latest_filing_downloads_document(tmp_path):
    with _mock_client() as client:
        filing, path = fetch_latest_filing("AAPL", form="10-Q", dest_dir=tmp_path, client=client)

    assert isinstance(filing, FilingRef)
    assert path == tmp_path / "aapl-20250927.htm"
    assert path.read_bytes() == b"<html>filing body</html>"
