"""
Generate a tiny SYNTHETIC sample filing PDF for local demos and tests.

The numbers in this document are invented for pipeline testing — it is
not a real SEC filing and must never be quoted as financial data. Its
only job is to exercise the parser (prose chunking + table extraction)
and the ingestion path without downloading a real 300-page 10-K.

Usage:
    python scripts/make_sample_filing.py [--out data/raw/sample_filing.pdf]
"""

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

PROSE = (
    "Management's Discussion and Analysis. Revenue grew in the quarter as "
    "services continued to expand faster than product sales. Gross margin "
    "benefited from a favourable mix shift toward higher-margin services, "
    "partly offset by higher component costs. Operating expenses rose "
    "modestly, driven by research and development headcount. The company "
    "generated strong operating cash flow and returned capital to "
    "shareholders through dividends and share repurchases. Liquidity "
    "remains strong, with cash and marketable securities comfortably "
    "exceeding current obligations. Looking ahead, management expects "
    "seasonal patterns consistent with prior years, while noting that "
    "macroeconomic conditions, foreign exchange movements and supply "
    "availability remain sources of uncertainty that could cause actual "
    "results to differ materially from expectations expressed here."
)


def build_sample_filing(out: Path) -> Path:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=letter,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    story = [
        Paragraph("SAMPLE CORP — Form 10-Q (SYNTHETIC SAMPLE)", styles["Title"]),
        Paragraph("Quarterly report for the period ended March 29, 2025. "
                  "All figures are invented for pipeline testing.", styles["Normal"]),
        Spacer(1, 16),
        Paragraph("Condensed Consolidated Statements of Operations (in millions)", styles["Heading2"]),
    ]

    table_data = [
        ["Line Item", "Q2 2025", "Q2 2024"],
        ["Total net sales", "$95,359", "$90,753"],
        ["Cost of sales", "$51,286", "$49,030"],
        ["Gross margin", "$44,073", "$41,723"],
        ["Research and development", "$8,550", "$7,907"],
        ["Total operating expenses", "$15,263", "$14,371"],
        ["Operating income", "$28,810", "$27,352"],
    ]
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    for _ in range(4):  # enough prose to exercise the chunker
        story.append(Paragraph(PROSE, styles["Normal"]))
        story.append(Spacer(1, 10))

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/sample_filing.pdf")
    args = parser.parse_args()
    path = build_sample_filing(Path(args.out))
    print(f"Wrote synthetic sample filing: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
