"""Turns raw document text into overlapping chunks that we can embed.

We split on token count rather than characters because the embedding model
cares about tokens, and a fixed character count gives wildly different chunk
sizes depending on the language / formatting of the doc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings


@dataclass
class Chunk:
    text: str
    source: str           # file name or Drive file id
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        # Stable id so re-ingesting the same file updates instead of duplicating.
        raw = f"{self.source}:{self.chunk_index}:{self.text[:64]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _splitter() -> RecursiveCharacterTextSplitter:
    s = get_settings()
    # token-based length so chunk_size actually means tokens
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer breaking on paragraph/sentence
    )


def chunk_document(text: str, source: str, extra_metadata: dict | None = None) -> list[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    pieces = _splitter().split_text(text)
    extra = extra_metadata or {}

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                text=piece,
                source=source,
                chunk_index=i,
                metadata={"source": source, "chunk_index": i, **extra},
            )
        )
    return chunks
