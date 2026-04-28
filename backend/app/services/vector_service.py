"""
Vector Service: Embed chunks via sentence-transformers (CPU) and store/query in ChromaDB.
"""

import uuid
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

# Model is loaded once at import time; first call downloads it (~90 MB) automatically.
_encoder: Optional[SentenceTransformer] = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


_chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=ChromaSettings(anonymized_telemetry=False),
)


def _get_collection():
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_text(text: str) -> List[float]:
    return _get_encoder().encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    return _get_encoder().encode(texts, normalize_embeddings=True).tolist()


def add_chunks(chunks: List[Dict]) -> int:
    """
    Add chunks to the vector store.
    Each chunk: {"text": str, "metadata": {"source": str, "page": int, "chunk_index": int}}
    """
    if not chunks:
        return 0

    collection = _get_collection()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [
        f"{c['metadata'].get('source','doc')}-p{c['metadata'].get('page',0)}-c{c['metadata'].get('chunk_index',0)}-{uuid.uuid4().hex[:8]}"
        for c in chunks
    ]
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


def query(question: str, top_k: int = TOP_K, source_filter: Optional[str] = None) -> List[Dict]:
    """Retrieve top-k relevant chunks for a question."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    q_emb = embed_text(question)
    where = {"source": source_filter} if source_filter else None

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, collection.count()),
        where=where,
    )

    out: List[Dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "metadata": meta or {}, "distance": float(dist)})
    return out


def list_documents() -> List[Dict]:
    """Return distinct sources currently indexed with chunk counts."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    res = collection.get(include=["metadatas"])
    counts: Dict[str, int] = {}
    for meta in res.get("metadatas") or []:
        if not meta:
            continue
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def delete_document(source: str) -> int:
    """Delete all chunks for a given source. Returns number removed."""
    collection = _get_collection()
    res = collection.get(where={"source": source})
    ids = res.get("ids") or []
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def reset_collection() -> None:
    """Drop and recreate the collection."""
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _get_collection()
