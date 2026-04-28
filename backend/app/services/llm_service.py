"""
LLM Service: Generate answers using Groq (free tier).
Models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma2-9b-it
"""

from typing import List, Dict, Iterator
from groq import Groq

from app.config import LLM_MODEL, GROQ_API_KEY


_client = Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided document context. If the answer is not in the context, say you "
    "don't know. Cite sources inline using [source p.PAGE] based on the "
    "metadata provided with each context chunk."
)


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[{i}] (source: {src} p.{page})\n{c.get('text', '').strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, context_chunks: List[Dict]) -> List[Dict]:
    context = _format_context(context_chunks) if context_chunks else "(no context retrieved)"
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above with inline citations like [source p.PAGE]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(question: str, context_chunks: List[Dict]) -> str:
    messages = build_messages(question, context_chunks)
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def generate_stream(question: str, context_chunks: List[Dict]) -> Iterator[str]:
    messages = build_messages(question, context_chunks)
    stream = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece
