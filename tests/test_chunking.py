"""
Tests for the prose chunker in src/parser.py and for parse_pdf running
end-to-end on a generated sample filing.

These tests are deliberately lightweight: they need only pdfplumber
(and reportlab to generate the fixture PDF) — no Qdrant, Groq, or
embedding-model downloads.
"""

import pytest

from src.parser import CHUNK_OVERLAP, CHUNK_SIZE, SECParser


@pytest.fixture()
def parser():
    return SECParser(ticker="AAPL", fiscal_year=2025, fiscal_period="Q2")


def test_short_text_is_one_unchanged_chunk(parser):
    text = "Revenue rose in the second quarter."
    assert parser._split_text_into_chunks(text) == [text]


def test_empty_text_gives_no_chunks(parser):
    assert parser._split_text_into_chunks("   \n  ") == []


def test_long_text_is_split_within_size_limit(parser):
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = parser._split_text_into_chunks(text)

    assert len(chunks) >= 4
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # Nothing is lost at either end of the document.
    assert "Sentence number 0." in chunks[0]
    assert "Sentence number 299." in chunks[-1]


def test_chunks_overlap(parser):
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = parser._split_text_into_chunks(text)

    # The start of the second chunk must already appear inside the first
    # chunk — that shared span is the overlap that keeps boundary facts whole.
    assert chunks[1][:40] in chunks[0]
    assert CHUNK_OVERLAP < CHUNK_SIZE  # overlap must never stall the splitter


def test_parse_pdf_chunks_long_page_and_tags_metadata(parser, tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    y = 750
    for i in range(45):  # ~45 lines x ~55 chars ≈ 2,400 chars on one page
        c.drawString(72, y, f"Line {i:02d}: Management discussed revenue, margins and outlook.")
        y -= 15
    c.showPage()
    c.drawString(72, 750, "Second page: risk factors and legal proceedings.")
    c.save()

    chunks = parser.parse_pdf(pdf_path)

    assert chunks, "expected chunks from a generated sample filing"
    for chunk in chunks:
        assert chunk.metadata["ticker"] == "AAPL"
        assert chunk.metadata["fiscal_year"] == 2025
        assert chunk.metadata["fiscal_period"] == "Q2"

    page1_text = [c for c in chunks if c.metadata["page"] == 1 and c.metadata["type"] == "text"]
    # The long first page must have been split, not stored as one chunk.
    assert len(page1_text) >= 2
    assert all(len(c.content) <= CHUNK_SIZE for c in page1_text)
    assert [c.metadata["chunk_index"] for c in page1_text] == list(range(len(page1_text)))
