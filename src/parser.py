"""
Document parser module.
Extracts text and tables from SEC filings (PDFs) and formats them into
structured, metadata-tagged chunks suitable for vector storage.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import pdfplumber


@dataclass
class DocumentChunk:
    """Represents an extracted piece of text or table with strict metadata."""
    content: str
    metadata: Dict[str, Any]


class SECParser:
    """Extracts raw text and markdown tables from SEC filing PDFs."""

    def __init__(self, ticker: str, fiscal_year: int, fiscal_period: str):
        self.ticker = ticker.upper()
        self.fiscal_year = fiscal_year
        self.fiscal_period = fiscal_period.upper()

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """Converts a raw 2D list of table cells into a clean Markdown table."""
        if not table or not any(table):
            return ""

        # Filter out completely empty rows and clean cell strings
        cleaned_rows = []
        for row in table:
            if not row:
                continue
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cleaned_row):
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return ""

        # Construct markdown table header and separator
        headers = cleaned_rows[0]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        # Construct data rows
        data_lines = []
        for row in cleaned_rows[1:]:
            # Ensure row length matches header length
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            data_lines.append("| " + " | ".join(row[:len(headers)]) + " |")

        return "\n".join([header_line, separator_line] + data_lines)

    def parse_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parses a PDF file, extracting text and tables page-by-page.
        Returns a list of structured DocumentChunk objects with metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        chunks: List[DocumentChunk] = []

        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # 1. Extract tables from the page first
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables, start=1):
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        chunks.append(
                            DocumentChunk(
                                content=md_table,
                                metadata={
                                    "ticker": self.ticker,
                                    "fiscal_year": self.fiscal_year,
                                    "fiscal_period": self.fiscal_period,
                                    "page": page_idx,
                                    "type": "table",
                                    "table_index": table_idx,
                                }
                            )
                        )

                # 2. Extract prose / narrative text
                text = page.extract_text()
                if text and text.strip():
                    chunks.append(
                        DocumentChunk(
                            content=text.strip(),
                            metadata={
                                "ticker": self.ticker,
                                "fiscal_year": self.fiscal_year,
                                "fiscal_period": self.fiscal_period,
                                "page": page_idx,
                                "type": "text",
                            }
                        )
                    )

        return chunks