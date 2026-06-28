#!/usr/bin/env python
"""Ingest local files into Pinecone from the command line.

Handy for backfilling existing docs or testing the pipeline without wiring up
n8n + Google Drive. In production the n8n workflow calls the same code path.

Usage:
    python ingest.py path/to/file.pdf
    python ingest.py path/to/folder/          # ingests every supported file
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag.loaders import load_text
from rag.pipeline import ingest_document

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def iter_files(target: Path):
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    target = Path(argv[0])
    if not target.exists():
        print(f"Path not found: {target}")
        return 1

    total_chunks = 0
    for file in iter_files(target):
        try:
            text = load_text(file)
        except ValueError as e:
            print(f"  skip  {file.name}: {e}")
            continue

        n = ingest_document(text, source=file.name, metadata={"path": str(file)})
        total_chunks += n
        print(f"  ok    {file.name}: {n} chunks")

    print(f"\nDone. {total_chunks} chunks indexed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
