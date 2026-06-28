"""Pull plain text out of the file types we actually get from Drive: pdf, docx,
txt/md, and Google Docs (which n8n exports as text before sending here).

Kept deliberately simple. If you need OCR or tables, this is where you'd plug it in.
"""

from __future__ import annotations

from pathlib import Path


def load_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix in {".docx"}:
        return _load_docx(path)

    raise ValueError(f"Unsupported file type: {suffix} ({path.name})")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    import docx  # python-docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)
