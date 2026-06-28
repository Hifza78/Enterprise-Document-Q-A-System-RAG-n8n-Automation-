"""The two things this whole project does: ingest a document, and answer a question.

Both n8n and the FastAPI app call into these functions so the logic lives in one
place. n8n handles the plumbing (Drive triggers, Telegram, scheduling); the
actual RAG work happens here.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .chunking import chunk_document
from .config import get_settings
from .prompts import build_user_message, system_prompt
from .vectorstore import Match, VectorStore

_store: VectorStore | None = None


def store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_store() -> None:
    """Drop the cached Pinecone connection. Call this after the keys change so
    the next request reconnects with the new credentials."""
    global _store
    _store = None


def ingest_document(text: str, source: str, metadata: dict | None = None, replace: bool = True) -> int:
    """Chunk + embed + upsert a single document. Returns number of chunks stored."""
    s = store()
    if replace:
        # Drop the old version first so edits don't leave stale chunks behind.
        s.delete_by_source(source)

    chunks = chunk_document(text, source=source, extra_metadata=metadata)
    return s.upsert(chunks)


@dataclass
class Answer:
    text: str
    sources: list[str]
    matches: list[Match]

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "sources": self.sources,
            "matches": [
                {"source": m.source, "score": round(m.score, 3), "chunk_index": m.chunk_index}
                for m in self.matches
            ],
        }


def answer_question(question: str, top_k: int | None = None) -> Answer:
    settings = get_settings()
    matches = store().search(question, top_k=top_k)

    if not matches:
        return Answer(
            text="I couldn't find this in the documentation.",
            sources=[],
            matches=[],
        )

    # Grok speaks the OpenAI API; just point the SDK at xAI's base URL.
    client = OpenAI(api_key=settings.chat_api_key, base_url=settings.chat_base_url)
    resp = client.chat.completions.create(
        model=settings.chat_model,
        temperature=0.1,  # keep it grounded, not creative
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": build_user_message(question, matches)},
        ],
    )

    # de-dupe sources but keep order
    seen, sources = set(), []
    for m in matches:
        if m.source not in seen:
            seen.add(m.source)
            sources.append(m.source)

    return Answer(text=resp.choices[0].message.content.strip(), sources=sources, matches=matches)
