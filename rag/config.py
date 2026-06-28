"""Central place for configuration.

Keys can come from two places:
  1. The .env file / environment (the usual way), or
  2. The web UI, which saves them back into .env at runtime.

Because of (2) the settings are cached but the cache can be cleared, so a key
entered in the browser takes effect without restarting the server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv, set_key

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    # Chat (answering) runs on xAI/Grok via the OpenAI-compatible API.
    chat_api_key: str = ""
    chat_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Pinecone
    index_name: str = "enterprise-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Models
    # Embeddings run locally via fastembed (no API key, no quota). bge-small is
    # 384 dims; if you change this, recreate the Pinecone index to match.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chat_model: str = "gemini-2.5-flash"  # free on Google AI Studio's tier

    # Retrieval / chunking
    chunk_size: int = 800          # tokens
    chunk_overlap: int = 120       # tokens
    top_k: int = 5
    # Drop weak matches so we don't feed junk to the LLM. Tuned for bge-small
    # cosine scores, which run lower than OpenAI's (a strong match is ~0.6-0.8).
    score_threshold: float = 0.5

    # Optional: Telegram bot for the chat interface
    telegram_token: str = ""


_cached: Settings | None = None


def _build() -> Settings:
    return Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        # Chat runs on any OpenAI-compatible endpoint (Groq, xAI, OpenAI, ...).
        # Pick up whichever provider key is set, in priority order.
        chat_api_key=(os.getenv("CHAT_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
                      or os.getenv("GROQ_API_KEY", "") or os.getenv("XAI_API_KEY", "")
                      or os.getenv("OPENAI_API_KEY", "")),
        chat_base_url=os.getenv("CHAT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        index_name=os.getenv("PINECONE_INDEX", "enterprise-docs"),
        chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        top_k=int(os.getenv("TOP_K", "5") or "5"),
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    )


def get_settings(require: bool = True) -> Settings:
    """Return current settings. With require=True (the default) this raises a
    clear error if the API keys are missing, which the API endpoints turn into a
    friendly 'add your keys' message."""
    global _cached
    if _cached is None:
        _cached = _build()

    if require:
        # Embeddings are local now, so the only keys we need are Grok (chat) and
        # Pinecone (storage).
        missing = [
            name
            for name, val in (("xAI/Grok", _cached.chat_api_key), ("Pinecone", _cached.pinecone_api_key))
            if not val
        ]
        if missing:
            raise MissingKeysError(missing)

    return _cached


def is_configured() -> bool:
    s = get_settings(require=False)
    return bool(s.chat_api_key and s.pinecone_api_key)


def update_keys(xai_api_key: str | None = None, pinecone_api_key: str | None = None,
                chat_model: str | None = None, index_name: str | None = None) -> None:
    """Persist keys to .env and refresh the in-memory settings so the change
    takes effect immediately. Called by the web UI's setup panel."""
    global _cached

    if not _ENV_PATH.exists():
        _ENV_PATH.touch()

    updates = {
        "XAI_API_KEY": xai_api_key,
        "PINECONE_API_KEY": pinecone_api_key,
        "CHAT_MODEL": chat_model,
        "PINECONE_INDEX": index_name,
    }
    for key, value in updates.items():
        if value:
            os.environ[key] = value
            set_key(str(_ENV_PATH), key, value)

    _cached = None  # force rebuild on next get_settings()


class MissingKeysError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"Missing API key(s): {', '.join(missing)}. "
            "Open the web UI and add them in the Setup panel, or set them in .env."
        )
