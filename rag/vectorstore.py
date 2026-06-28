"""Pinecone access. Creates the index on first run if it isn't there yet."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

from .chunking import Chunk
from .config import get_settings
from .embeddings import embed_texts

_EMBED_DIMS = 384  # BAAI/bge-small-en-v1.5 (local fastembed)


@dataclass
class Match:
    text: str
    source: str
    score: float
    chunk_index: int


class VectorStore:
    def __init__(self) -> None:
        # Storage + embeddings don't need the chat key, so don't require it here;
        # only answering enforces the Grok key.
        self.settings = get_settings(require=False)
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.settings.index_name)

    def _ensure_index(self) -> None:
        name = self.settings.index_name
        existing = {ix["name"] for ix in self.pc.list_indexes()}

        # If an index exists at the wrong dimension (e.g. left over from the old
        # 1536-dim OpenAI embeddings), drop it so we can recreate it at 384.
        if name in existing:
            if self.pc.describe_index(name).dimension != _EMBED_DIMS:
                self.pc.delete_index(name)
                existing.discard(name)

        if name not in existing:
            self.pc.create_index(
                name=name,
                dimension=_EMBED_DIMS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            # create_index returns before the index is queryable; wait for it.
            for _ in range(60):
                if self.pc.describe_index(name).status.get("ready"):
                    break
                time.sleep(1)

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        vectors = embed_texts([c.text for c in chunks])
        records = []
        for chunk, vec in zip(chunks, vectors):
            records.append(
                {
                    "id": chunk.id,
                    "values": vec,
                    # Pinecone metadata has a size cap, so store a trimmed copy of the text.
                    "metadata": {**chunk.metadata, "text": chunk.text[:4000]},
                }
            )

        # Upsert in batches; Pinecone rejects very large payloads.
        for start in range(0, len(records), 100):
            self.index.upsert(vectors=records[start : start + 100])

        return len(records)

    def search(self, query: str, top_k: int | None = None) -> list[Match]:
        from .embeddings import embed_query

        top_k = top_k or self.settings.top_k
        result = self.index.query(
            vector=embed_query(query),
            top_k=top_k,
            include_metadata=True,
        )

        matches: list[Match] = []
        for m in result.get("matches", []):
            if m["score"] < self.settings.score_threshold:
                continue  # too weak, skip rather than feed noise to the LLM
            md = m.get("metadata", {})
            matches.append(
                Match(
                    text=md.get("text", ""),
                    source=md.get("source", "unknown"),
                    score=float(m["score"]),
                    chunk_index=int(md.get("chunk_index", 0)),
                )
            )
        return matches

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks for a file (used when a Drive file is deleted/replaced)."""
        try:
            self.index.delete(filter={"source": source})
        except NotFoundException:
            # Nothing indexed yet: a fresh index has no default namespace, so a
            # filtered delete 404s. There's nothing to remove, so treat as a no-op.
            pass
