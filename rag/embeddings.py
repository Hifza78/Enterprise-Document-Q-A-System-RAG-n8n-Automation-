"""Local embeddings via fastembed — no API key, no quota, no network call.

fastembed runs a small ONNX model on the CPU. The model is downloaded once on
first use and cached under the user's home directory. We default to
BAAI/bge-small-en-v1.5 (384 dims); the Pinecone index must match that size.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from .config import get_settings

_BATCH = 256  # fastembed batches internally; this just caps memory per call


@lru_cache(maxsize=2)
def _model(name: str) -> TextEmbedding:
    # Loading the model is the expensive part, so cache it per model name.
    return TextEmbedding(model_name=name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, preserving order."""
    if not texts:
        return []

    model = _model(get_settings().embed_model)
    # .embed() yields numpy arrays in input order; convert to plain float lists
    # so they serialize cleanly into Pinecone.
    return [vec.tolist() for vec in model.embed(list(texts), batch_size=_BATCH)]


def embed_query(text: str) -> list[float]:
    """Embed a single search query. bge models are trained with a query-specific
    instruction, so use query_embed rather than the plain document path."""
    model = _model(get_settings().embed_model)
    return next(iter(model.query_embed([text]))).tolist()
