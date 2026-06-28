#!/usr/bin/env python
"""Quick terminal chat against the indexed docs. Good for sanity-checking
retrieval without spinning up the server.

    python query.py "how do I rotate the API keys?"
    python query.py            # interactive loop
"""

from __future__ import annotations

import sys

from rag.pipeline import answer_question


def ask(question: str) -> None:
    answer = answer_question(question)
    print("\n" + answer.text)
    if answer.sources:
        print("\nsources: " + ", ".join(answer.sources))
    print()


def main(argv: list[str]) -> int:
    if argv:
        ask(" ".join(argv))
        return 0

    print("Ask a question (Ctrl-C to quit).")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                ask(q)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
