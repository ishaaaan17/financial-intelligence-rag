# Data directory

Staging area for filings. Contents are git-ignored (except `.gitkeep`
files and this README) — filings are large and often fetched fresh.

## Layout

- `raw/` — source documents waiting to be ingested: real SEC PDFs you
  upload, filings downloaded from EDGAR, and the synthetic sample.
- `processed/` — reserved for derived artefacts (not currently written
  by the pipeline; chunks go straight to Qdrant).

## Getting a filing in here

Real filing, straight from SEC EDGAR:

```bash
python scripts/fetch_filing.py --ticker AAPL --form 10-Q
```

Synthetic sample (invented numbers, only for exercising the pipeline):

```bash
python scripts/make_sample_filing.py
# -> data/raw/sample_filing.pdf
```

Then ingest a PDF via the API (`POST /ingest`) or the Streamlit UI.
