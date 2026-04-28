"""
Document store + retrieval service.

Uses BM25 (keyword-based ranking) for retrieval — no embedding model
required, no internet, no GPU. Documents are persisted as JSON to disk.
"""

import json
import re
import threading
import uuid
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.config import CHROMA_DIR, TOP_K


_STORE_FILE = CHROMA_DIR / "store.json"
_lock = threading.Lock()

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _load() -> List[Dict]:
    if not _STORE_FILE.exists():
        return []
    try:
        return json.loads(_STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: List[Dict]) -> None:
    _STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STORE_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def add_chunks(chunks: List[Dict]) -> int:
    """Persist chunks. Each: {"text": str, "metadata": {"source","page","chunk_index"}}"""
    if not chunks:
        return 0
    with _lock:
        items = _load()
        for c in chunks:
            items.append(
                {
                    "id": uuid.uuid4().hex,
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                }
            )
        _save(items)
    return len(chunks)


def query(question: str, top_k: int = TOP_K, source_filter: Optional[str] = None) -> List[Dict]:
    """Retrieve top-k relevant chunks via BM25."""
    items = _load()
    if source_filter:
        items = [it for it in items if it.get("metadata", {}).get("source") == source_filter]
    if not items:
        return []

    corpus = [_tokenize(it["text"]) for it in items]
    bm25 = BM25Okapi(corpus)

    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    indices = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "text": items[i]["text"],
            "metadata": items[i].get("metadata", {}),
            "distance": float(-scores[i]),
        }
        for i in indices
        if scores[i] > 0
    ]


def list_documents() -> List[Dict]:
    items = _load()
    counts: Dict[str, int] = {}
    for it in items:
        src = it.get("metadata", {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def delete_document(source: str) -> int:
    with _lock:
        items = _load()
        before = len(items)
        items = [it for it in items if it.get("metadata", {}).get("source") != source]
        removed = before - len(items)
        if removed:
            _save(items)
    return removed


def reset_collection() -> None:
    with _lock:
        _save([])
