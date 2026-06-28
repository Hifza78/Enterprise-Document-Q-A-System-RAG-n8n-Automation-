"""Boots the app logic in-process (no server, no real API keys) and checks the
endpoints behave. Run: python scripts/smoketest.py

This calls the route functions directly so it doesn't depend on a particular
starlette/httpx TestClient version."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `rag` and `app` import

# Redirect key persistence to a throwaway .env so we never touch a real one.
from rag import config as _config

_config._ENV_PATH = Path(tempfile.gettempdir()) / "rag_smoketest.env"
if _config._ENV_PATH.exists():
    _config._ENV_PATH.unlink()
_config._cached = None  # start from a clean, unconfigured state

import app as application
from app import ChatRequest, ConfigRequest
from rag.config import MissingKeysError


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


# UI file is served
resp = application.index()
check("GET / serves the dashboard file", Path(resp.path).name == "index.html" and Path(resp.path).exists())

# status works with no keys
st = application.status()
check("status reports not-configured before keys", st["configured"] is False)

# asking with no keys raises the friendly MissingKeysError (turned into 428 by the handler)
raised = False
try:
    application.chat(ChatRequest(question="hi"))
except MissingKeysError as e:
    raised = "key" in str(e).lower()
check("chat without keys raises MissingKeysError", raised)

# saving keys flips status to configured
st2 = application.set_config(ConfigRequest(xai_api_key="xai-test", pinecone_api_key="pcn-test"))
check("set_config saves keys and reports configured", st2["configured"] and st2["has_xai"] and st2["has_pinecone"])

# cleanup
if _config._ENV_PATH.exists():
    _config._ENV_PATH.unlink()

print("\nAll smoke tests passed.")
