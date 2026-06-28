"""Builds the prompt we send to the chat model.

The whole point of RAG is that the answer is grounded in retrieved text, so the
context block is formatted to make citation easy: each passage gets a [source]
tag and the system prompt tells the model to reuse those tags verbatim.
"""

from __future__ import annotations

from pathlib import Path

from .vectorstore import Match

_SYSTEM_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt").read_text(
    encoding="utf-8"
)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_context_block(matches: list[Match]) -> str:
    if not matches:
        return "(no relevant passages found)"

    blocks = []
    for m in matches:
        blocks.append(f"[{m.source}] (relevance {m.score:.2f})\n{m.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, matches: list[Match]) -> str:
    context = build_context_block(matches)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite the [source] tags."
    )
