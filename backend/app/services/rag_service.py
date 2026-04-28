"""
RAG Service: Orchestrate ingestion (PDF -> chunks -> vectors) and Q&A.
"""

from pathlib import Path
from typing import Dict, List, Optional

from app.config import UPLOAD_DIR, TOP_K
from app.services import pdf_service, vector_service, llm_service


def ingest_pdf(pdf_path: str, filename: str) -> Dict:
    """Process a PDF (already saved to disk) into the vector store."""
    chunks = pdf_service.process_pdf(pdf_path, filename)
    added = vector_service.add_chunks(chunks)
    return {
        "filename": filename,
        "chunks_added": added,
        "pages": len({c["metadata"]["page"] for c in chunks}) if chunks else 0,
    }


def save_and_ingest(file_bytes: bytes, filename: str) -> Dict:
    """Persist uploaded bytes to disk and ingest."""
    safe_name = Path(filename).name
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return ingest_pdf(str(dest), safe_name)


def answer(
    question: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> Dict:
    """Retrieve context and generate an answer with sources."""
    chunks = vector_service.query(question, top_k=top_k, source_filter=source_filter)

    if not chunks:
        return {
            "answer": "I don't have any indexed documents to answer from. Please upload a PDF first.",
            "sources": [],
        }

    text = llm_service.generate(question, chunks)
    sources = _format_sources(chunks)
    return {"answer": text, "sources": sources}


def _format_sources(chunks: List[Dict]) -> List[Dict]:
    out = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        out.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "snippet": (c.get("text") or "")[:240],
                "distance": c.get("distance"),
            }
        )
    return out
