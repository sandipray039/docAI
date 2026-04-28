"""
PDF Service: Extract text from PDFs and split into chunks for retrieval.
"""

import fitz
from typing import List, Dict
from pathlib import Path

from app.config import CHUNK_SIZE, CHUNK_OVERLAP,UPLOAD_DIR


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text page-by-page from a PDF."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1): # type: ignore[arg-type]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_number": page_num,
                "text": text.strip()
            })

    doc.close()
    return pages


def get_pdf_info(pdf_path: str) -> Dict:
    """Get lightweight PDF metadata."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    doc = fitz.open(pdf_path)
    info = {
        "page_count": doc.page_count,
        "metadata": doc.metadata,
        "file_size_bytes": Path(pdf_path).stat().st_size
    }
    doc.close()
    return info


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks using a sliding window."""
    if len(text) <= chunk_size:
        return [text]

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks


def process_pdf(pdf_path: str, filename: str) -> List[Dict]:
    """
    Full pipeline: extract pages → chunk each page → attach metadata.

    Args:
        pdf_path: path to PDF on disk
        filename: original filename to store in metadata for citations

    Returns list of dicts:
        [
          {
            "text": "...",
            "metadata": {
              "source": "test.pdf",
              "page": 1,
              "chunk_index": 0
            }
          },
          ...
        ]
    """
    resolved = Path(pdf_path)
    if not resolved.is_absolute():
        resolved = UPLOAD_DIR / resolved.name

    pages = extract_text_from_pdf(str(resolved)) 
    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for i, chunk_value in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk_value,
                "metadata": {
                    "source": filename,
                    "page": page["page_number"],
                    "chunk_index": i,
                }
            })

    return all_chunks