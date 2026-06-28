"""FastAPI app: serves the web UI and exposes the RAG pipeline over HTTP.

Open http://localhost:8000 in a browser for the dashboard (setup, ingest, chat,
and a live view of the pipeline). The same endpoints are what the n8n workflows
call.

Endpoints:
    GET  /                 -> the web dashboard
    GET  /api/status       -> are the keys set? which models?
    POST /api/config       -> save OpenAI / Pinecone keys
    POST /chat             -> {"question": "..."}            ask a question
    POST /ingest           -> {"text": "...", "source": "..."} push raw text in
    POST /api/upload       -> multipart file upload (pdf/docx/txt/md)
    POST /telegram         -> Telegram webhook update
    GET  /healthz          -> health check
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config
from rag.config import MissingKeysError, get_settings
from rag.loaders import load_text
from rag.pipeline import answer_question, ingest_document, reset_store

app = FastAPI(title="Enterprise Document Q&A")

WEB_DIR = Path(__file__).resolve().parent / "web"


# --- request models ----------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class IngestRequest(BaseModel):
    text: str
    source: str
    metadata: Optional[dict] = None


class ConfigRequest(BaseModel):
    xai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    chat_model: Optional[str] = None
    index_name: Optional[str] = None


# --- error handling ----------------------------------------------------------

@app.exception_handler(MissingKeysError)
def _missing_keys_handler(_request, exc: MissingKeysError):
    # 428 = "Precondition Required"; the UI uses this to nudge you to Setup.
    return JSONResponse(status_code=428, content={"error": str(exc), "missing": exc.missing})


def _explain(e: Exception) -> str:
    """Turn a raw provider exception into a one-line message the UI can show."""
    msg = str(e)
    if "insufficient_quota" in msg or "exceeded your current quota" in msg or "rate limit" in msg.lower():
        return ("The chat provider rejected the request: out of quota or rate-limited. "
                "Wait a moment and retry, or check the provider account.")
    if "invalid_api_key" in msg or "Incorrect API key" in msg or "Invalid API Key" in msg or "401" in msg:
        return "The chat provider rejected the API key as invalid. Check the Gemini key in Setup."
    if "permission-denied" in msg or "credits or licenses" in msg or "403" in msg:
        return "The chat key is valid but the account has no credits/access. Add credits with the provider, then retry."
    if "does not exist" in msg or ("model" in msg.lower() and "not found" in msg.lower()) or "decommissioned" in msg:
        return "That chat model id isn't available. Pick another in Setup (e.g. gemini-2.0-flash)."
    if "Namespace not found" in msg:
        return "Pinecone namespace not found (index is empty) — nothing to do."
    return f"{type(e).__name__}: {msg[:300]}"


# --- web UI ------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# --- status & config ---------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    s = get_settings(require=False)
    return {
        "configured": config.is_configured(),
        "has_xai": bool(s.chat_api_key),
        "has_pinecone": bool(s.pinecone_api_key),
        "chat_model": s.chat_model,
        "embed_model": s.embed_model,
        "index_name": s.index_name,
        "top_k": s.top_k,
    }


@app.post("/api/config")
def set_config(req: ConfigRequest):
    config.update_keys(
        xai_api_key=req.xai_api_key,
        pinecone_api_key=req.pinecone_api_key,
        chat_model=req.chat_model,
        index_name=req.index_name,
    )
    reset_store()  # reconnect Pinecone with the new credentials on next call
    return status()


# --- RAG endpoints -----------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        answer = answer_question(req.question, top_k=req.top_k)
    except MissingKeysError:
        raise
    except Exception as e:  # surface the real reason instead of a bare 500
        return JSONResponse(status_code=502, content={"error": _explain(e)})
    return answer.to_dict()


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        n = ingest_document(req.text, source=req.source, metadata=req.metadata)
    except Exception as e:  # surface the real reason instead of a bare 500
        return JSONResponse(status_code=502, content={"error": _explain(e)})
    return {"source": req.source, "chunks_indexed": n}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    data = await file.read()

    # load_text works off a path, so stage the upload in a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text = load_text(tmp_path)
    except ValueError as e:
        return JSONResponse(status_code=415, content={"error": str(e)})
    finally:
        os.unlink(tmp_path)

    try:
        n = ingest_document(text, source=file.filename, metadata={"upload": True})
    except Exception as e:  # surface the real reason instead of a bare 500
        return JSONResponse(status_code=502, content={"error": _explain(e)})
    return {"source": file.filename, "chunks_indexed": n, "chars": len(text)}


# --- Telegram chat interface -------------------------------------------------

@app.post("/telegram")
async def telegram_webhook(update: dict):
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    answer = answer_question(text)
    reply = _format_for_telegram(answer)
    await _send_telegram(chat_id, reply)
    return {"ok": True}


def _format_for_telegram(answer) -> str:
    if not answer.sources:
        return answer.text
    cited = ", ".join(answer.sources)
    return f"{answer.text}\n\nSources: {cited}"


async def _send_telegram(chat_id: int, text: str) -> None:
    token = get_settings(require=False).telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})
