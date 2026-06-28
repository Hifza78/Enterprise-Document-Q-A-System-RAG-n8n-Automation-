# Enterprise Document Q&A (RAG)

Ask questions about your company's documents and get back an answer that tells
you where it came from. Drop a file into a folder (or a Google Drive folder),
wait a few seconds, and you can ask about it from a web page, a webhook, or
Telegram.

I built this because "just search the wiki" falls apart once you have a few
hundred documents nobody remembers the names of. Keyword search finds the page;
it doesn't answer the question. This does both — it pulls the relevant passages
and has an LLM write the answer, with the source filenames attached so you can
go check it yourself.

## How it works

Two pieces. **n8n** handles the plumbing (Drive triggers, scheduling, Telegram,
webhooks) and a small **Python service** does the actual RAG work. I kept the
logic in Python on purpose so I could test it from a terminal instead of
clicking through n8n every time I changed something.

**Ingesting a document** (happens automatically when a file shows up in Drive):

```
file lands in Drive  ->  n8n trigger  ->  download + extract text
                                           -> POST /ingest
                                              -> chunk -> embed -> store in Pinecone
```

**Answering a question:**

```
webhook / Telegram  ->  n8n  ->  POST /chat
                                  -> embed the question
                                  -> find the closest passages in Pinecone
                                  -> build a prompt from them
                                  -> LLM writes the answer + citations
```

## Stack

- **Python + FastAPI** — the ingest/answer service ([app.py](app.py))
- **fastembed** — runs the embedding model locally, so there's no embedding API
  key and nothing to pay for. Default is `BAAI/bge-small-en-v1.5` (384 dims).
- **Pinecone** — serverless vector index for the chunks
- **Gemini** — `gemini-2.5-flash` on Google's free tier writes the answers. The
  chat layer talks to any OpenAI-compatible endpoint, so swapping to OpenAI,
  Groq, or xAI is just a `.env` change (`CHAT_BASE_URL` + key + model).
- **n8n** — Drive trigger, scheduling, Telegram, webhook glue
- **LangChain** — only the token-aware text splitter, nothing heavier

## Running it

The easiest way to use the whole thing is the built-in web page. You don't need
n8n or Google Drive to start — paste your keys into the browser and go.

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** and you get one page with four parts:

1. **Setup** — paste your Gemini and Pinecone keys. They're saved to a local
   `.env` on the server (never sent anywhere else) and take effect right away,
   no restart. Embeddings run locally, so they don't need a key.
2. **Pipeline view** — a little diagram that lights up step by step every time
   you ask something, so you can actually see what RAG is doing.
3. **Add documents** — drag in a pdf / docx / txt / md, or paste text.
4. **Ask** — a chat box that shows the answer, the source files it cited, and
   the exact passages it retrieved (with similarity scores).

### Where the keys come from

- **Gemini** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  → "Create API key" (starts with `AIza`). The free tier is enough.
- **Pinecone** — [app.pinecone.io](https://app.pinecone.io) → API Keys. The free
  Starter tier works, and the index is created for you on first use.

You can also skip the UI and put the keys in `.env` (`cp .env.example .env`).
Both paths feed the same config.

### From the terminal

Everything the UI does also works from the command line:

```bash
python ingest.py data/sample-handbook.md
python query.py "how long does a laptop request take?"
```

You should get back something like *"New-hire laptop requests typically take 3
business days from approval [sample-handbook.md]."*

### Wiring up n8n

Import the two workflows from [n8n/](n8n/), add your Google Drive and
(optionally) Telegram credentials, and set the Drive folder ID in the trigger
node. The HTTP nodes default to `http://localhost:8000` — change that if the
service runs somewhere else.

## A few decisions worth explaining

**Chunking by tokens, not characters.** A 1,000-character chunk can be 150
tokens or 400 depending on the text, which makes retrieval inconsistent. The
splitter counts tokens and prefers to break on paragraph/sentence boundaries so
chunks stay whole. Defaults are 800 tokens with 120 overlap — see
[rag/config.py](rag/config.py) to tune it.

**A relevance threshold before the LLM sees anything.** If the best match scores
too low it gets dropped, and if nothing clears the bar the answer is just *"I
couldn't find this in the documentation."* Feeding weak matches to the model was
where most of the made-up answers came from, so this did more for reliability
than any prompt tweak. (The threshold is tuned for the local embedding model,
whose cosine scores run a bit lower than the hosted ones.)

**Stable chunk IDs.** Each chunk's ID is a hash of its source + position + text,
so re-ingesting an edited file updates the existing vectors instead of piling up
duplicates. The pipeline also deletes a file's old chunks before re-indexing it.

**Citations are part of the deal.** The system prompt
([prompts/system_prompt.txt](prompts/system_prompt.txt)) tells the model to
answer only from the passages it's given and to reuse the `[source]` tags
exactly. Temperature stays low. It's not bulletproof, but combined with the
retrieval threshold it's the difference between "sounds plausible" and "actually
grounded."

**Local embeddings.** Embeddings run on the machine with fastembed instead of a
paid API. No key, no quota, no per-request cost — and ingestion still works if
you're offline. The trade-off is a one-time model download (~70 MB) on first
use, and recreating the Pinecone index if you switch to an embedder with a
different dimension (the code handles that automatically).

## Layout

```
app.py                  FastAPI service: web UI + /chat, /ingest, /api/*, /telegram
web/index.html          the dashboard (setup, pipeline view, ingest, chat)
ingest.py               CLI to index local files
query.py                CLI to ask questions
rag/
  config.py             settings + runtime key updates from the UI
  chunking.py           token-aware splitting
  embeddings.py         local fastembed embeddings
  vectorstore.py        Pinecone: create index, upsert, search
  prompts.py            builds the grounded prompt
  pipeline.py           ingest_document() and answer_question()
  loaders.py            pdf / docx / txt / md -> text
n8n/                    importable workflow JSON
prompts/                system prompt
scripts/                eval harness, question set, smoke test
data/                   sample doc to test with
```

## Things I'd do next

- Re-rank the retrieved chunks (cross-encoder) before the LLM sees them — top-k
  by cosine similarity is fine, but a re-ranker noticeably helps on edge cases.
- Show the snippet behind each citation in the reply, not just the filename.
- Per-user access control on the index so people only retrieve docs they're
  allowed to see. Right now it assumes one trusted corpus.
- Move the in-process ingestion onto a queue if the volume grows.

## Notes

`.env` is gitignored — don't commit your keys. The Drive trigger polls once a
minute in the sample workflow, which is fine for most cases; n8n can do push
notifications if you need near-instant sync.
