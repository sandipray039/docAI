"""
HTTP routes for the RAG API.
"""

from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import TOP_K
from app.services import rag_service, vector_service, llm_service


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = TOP_K
    source: Optional[str] = None


class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    pages: int


class DocumentItem(BaseModel):
    source: str
    chunks: int


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        result = rag_service.save_and_ingest(data, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest: {e}")
    return result


@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        result = rag_service.answer(req.question, top_k=req.top_k, source_filter=req.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    return result


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    """Stream tokens as they are generated. Sources come back as a trailing JSON line."""
    try:
        chunks = vector_service.query(req.question, top_k=req.top_k, source_filter=req.source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    def gen():
        if not chunks:
            yield "I don't have any indexed documents to answer from. Please upload a PDF first."
            return
        for piece in llm_service.generate_stream(req.question, chunks):
            yield piece

    return StreamingResponse(gen(), media_type="text/plain")


@router.get("/documents", response_model=List[DocumentItem])
def list_docs():
    return vector_service.list_documents()


@router.delete("/documents/{source}")
def delete_doc(source: str):
    removed = vector_service.delete_document(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Source not found.")
    return {"source": source, "chunks_removed": removed}


@router.post("/reset")
def reset():
    vector_service.reset_collection()
    return {"status": "reset"}
