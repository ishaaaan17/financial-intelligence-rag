"""
SEC EDGAR client.

Fetches real filings from EDGAR so ingestion is not limited to manually
uploaded PDFs: resolve a ticker to its CIK, find the latest filing of a
given form (10-K / 10-Q / ...), and download the primary document.

SEC fair-access rules require a declared User-Agent with contact info;
set the EDGAR_USER_AGENT environment variable (e.g.
"MyProject/1.0 (research; me@example.com)"). All requests go through one
httpx client with that header and a bounded timeout.

Only stdlib + httpx are used, and every function accepts an injectable
httpx.Client so tests can run against httpx.MockTransport with no network.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_USER_AGENT = "SEC-Financial-Intelligence-RAG/1.1 (research project)"


class EdgarError(RuntimeError):
    """Raised when EDGAR cannot resolve or fetch what was asked for."""


@dataclass
class FilingRef:
    """Identifies one filing in EDGAR."""
    ticker: str
    cik: str
    form: str
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def document_url(self) -> str:
        accession_nodash = self.accession_number.replace("-", "")
        return f"{ARCHIVES_URL}/{int(self.cik)}/{accession_nodash}/{self.primary_document}"


def _make_client(client: Optional[httpx.Client] = None) -> httpx.Client:
    if client is not None:
        return client
    user_agent = os.environ.get("EDGAR_USER_AGENT", DEFAULT_USER_AGENT)
    return httpx.Client(
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=30.0,
        follow_redirects=True,
    )


def _get_json(client: httpx.Client, url: str) -> dict:
    try:
        response = client.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise EdgarError(f"EDGAR request failed for {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise EdgarError(f"EDGAR returned non-JSON for {url}: {e}") from e


def get_cik(ticker: str, client: Optional[httpx.Client] = None) -> str:
    """Resolves a ticker symbol to its zero-padded 10-digit CIK."""
    owns_client = client is None
    client = _make_client(client)
    try:
        data = _get_json(client, TICKERS_URL)
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        raise EdgarError(f"Ticker not found in EDGAR company list: {ticker}")
    finally:
        if owns_client:
            client.close()


def find_latest_filing(submissions: dict, form: str, ticker: str = "", cik: str = "") -> FilingRef:
    """
    Picks the most recent filing of ``form`` from a submissions payload
    (the ``filings.recent`` block is ordered newest-first by EDGAR).
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms: List[str] = recent.get("form", [])
    for i, entry_form in enumerate(forms):
        if entry_form.upper() == form.upper():
            return FilingRef(
                ticker=ticker.upper(),
                cik=cik,
                form=entry_form,
                accession_number=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent.get("reportDate", [""])[i],
                primary_document=recent["primaryDocument"][i],
            )
    raise EdgarError(f"No {form} filing found in recent submissions")


def get_latest_filing(
    ticker: str,
    form: str = "10-Q",
    client: Optional[httpx.Client] = None,
) -> FilingRef:
    """ticker → CIK → submissions → latest filing of the given form."""
    owns_client = client is None
    client = _make_client(client)
    try:
        cik = get_cik(ticker, client=client)
        submissions = _get_json(client, SUBMISSIONS_URL.format(cik=cik))
        return find_latest_filing(submissions, form=form, ticker=ticker, cik=cik)
    finally:
        if owns_client:
            client.close()


def download_filing(
    filing: FilingRef,
    dest_dir: Path,
    client: Optional[httpx.Client] = None,
) -> Path:
    """Downloads the filing's primary document into ``dest_dir``."""
    owns_client = client is None
    client = _make_client(client)
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filing.primary_document
        try:
            response = client.get(filing.document_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise EdgarError(f"Failed to download filing {filing.document_url}: {e}") from e
        dest.write_bytes(response.content)
        return dest
    finally:
        if owns_client:
            client.close()


def fetch_latest_filing(
    ticker: str,
    form: str = "10-Q",
    dest_dir: Path = Path("data/raw"),
    client: Optional[httpx.Client] = None,
) -> tuple[FilingRef, Path]:
    """Convenience wrapper: resolve, find, and download in one call."""
    owns_client = client is None
    client = _make_client(client)
    try:
        filing = get_latest_filing(ticker, form=form, client=client)
        path = download_filing(filing, dest_dir=dest_dir, client=client)
        return filing, path
    finally:
        if owns_client:
            client.close()
