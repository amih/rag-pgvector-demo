# rag-pgvector-demo

**Phase 0 of the AI-security learning path: demystify RAG on your own turf.**

A tiny, dependency-light Retrieval-Augmented Generation demo that proves you don't need a vector database — a
vector search is just an index in the Postgres you already run. Everything is local and open-source: Postgres +
`pgvector` for storage/search, [Ollama](https://ollama.com) for embeddings and generation. One production
dependency (`pg`).

```
docs/*.md ──▶ chunk ──▶ embed (Ollama) ──▶ Postgres (pgvector)
                                                  │
question ──▶ embed ──▶ nearest chunks (cosine) ──▶ prompt ──▶ Ollama ──▶ answer
```

## Prerequisites

- **Docker** (for the Postgres+pgvector container) — or your own Postgres with the `vector` extension.
- **Node.js >= 20.6** (uses built-in `fetch` and `--env-file`).
- **Ollama** running locally, with the two models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings (768 dims)
  ollama pull llama3.1:8b        # answer generation
  ```
  (`llama3.1:8b` gives much better grounded answers than the tiny `llama3.2`. See
  [Getting better answers](#getting-better-answers).)

## Setup

```bash
cp .env.example .env       # defaults work out of the box
npm install
npm run db:up              # start Postgres+pgvector on localhost:5462
npm run schema             # create extension, table, HNSW index
```

## Use

```bash
npm run ingest                                   # load + embed everything in docs/
npm run query -- "What is a seam in the AI security stack?"
npm run query -- "Do I need a dedicated vector database?"
```

Drop your own `.md` / `.txt` files into `docs/` (e.g. your smallestbusiness.com posts) and re-run
`npm run ingest` to query your own writing.

## How it works

- **`src/schema.js`** — `CREATE EXTENSION vector`, a `chunks` table with a `vector(768)` column, and an HNSW
  cosine index. The dimension comes from `EMBED_DIM`; change the model → change the dim → re-run on a fresh DB.
- **`src/ingest.js`** — reads `docs/`, chunks each file (`src/chunk.js`), embeds each chunk (`src/embed.js`),
  and inserts text + vector.
- **`src/query.js`** — embeds your question, retrieves the `TOP_K` nearest chunks with
  `ORDER BY embedding <=> $query`, stuffs them into a grounded prompt, and generates the answer
  (`src/llm.js`). It prints the retrieved sources + similarity scores so you can see the retrieval working.

## Knobs (`.env`)

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgres://rag:rag@localhost:5462/rag` | Postgres connection |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base |
| `EMBED_MODEL` / `EMBED_DIM` | `nomic-embed-text` / `768` | embedding model + its dimension |
| `CHAT_MODEL` | `llama3.1:8b` | generation model |
| `TOP_K` | `5` | chunks retrieved per query |
| `NUM_CTX` | `8192` | generation context window (must fit passages + question) |
| `TEMPERATURE` | `0.2` | lower = more grounded |

## Getting better answers

Retrieval and generation are separate steps, and they fail separately. If the printed **similarity scores**
are high (~0.7+) but the **answer** is a shrug ("I don't know"), the retrieval was fine and the *generation*
is the problem. Usual causes, in order:

1. **The model is too small.** `llama3.2` (3B) often refuses even with the right passages in front of it. Use
   `CHAT_MODEL=llama3.1:8b` (default), or `qwen2.5:14b-instruct` / larger on a beefy machine.
2. **The context window is too small.** If the passages + question overflow `NUM_CTX`, Ollama truncates the
   prompt and the model literally can't see the question. Keep `NUM_CTX` ≥ 8192 for `TOP_K=5`.
3. **Too few / too many chunks.** Tune `TOP_K`. More chunks = more coverage but more noise (and more context).

If instead the **similarity scores are low** (~0.4 and below), that's a *retrieval* miss — the answer isn't in
your docs, or your chunks are too big/small. Tune the chunker in `src/chunk.js`.

## Reset

```bash
npm run db:reset && npm run schema   # wipe the volume and start clean
```

## Where this goes next

This is a starting point. The natural next step is to put a security gateway in front of the model —
prompt-injection defense, PII redaction, output validation, spend caps, and tracing — so the same retrieval
flow runs *governed*. See [smallestbusiness/guardrail](https://github.com/smallestbusiness/guardrail) for the
fail-closed AI gateway that pattern builds on.

---

Built by [Ami Heines](https://amiheines.com) · more at [smallestbusiness.com](https://smallestbusiness.com).
