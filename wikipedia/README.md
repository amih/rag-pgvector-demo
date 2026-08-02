# Wikipedia RAG — full English Wikipedia into pgvector

Scales the [parent demo](../README.md) from a folder of `.md` files to the whole
English Wikipedia ZIM, on the home server's GPU. **Resumable**, so a multi-hour /
multi-day embedding run survives Ctrl-C, reboots, and power cuts.

```
ZIM entry ─▶ clean+trim ─▶ chunk(overlap) ─▶ embed(batched, GPU) ─▶ COPY → Postgres
                                                                     + cursor bump (same txn)
question ─▶ embed(search_query:) ─▶ nearest chunks (pgvector) ─▶ Ollama ─▶ answer
```

Written in Python (not the demo's Node) because the right tools here — `libzim`
for the ZIM and `sentence-transformers` for fast GPU embedding — are Python-native.

## Measured on this box (GTX 1080)

| Thing | Value |
|---|---|
| ZIM | `wikipedia_en_all_nopic_2026-06.zim`, 50 GB, 19.7M entries |
| Real content articles | **~6.9M** (libzim's `article_count` 19.2M *includes redirects*) |
| Chunks/article @1200 chars | 4.16 prose-clean · 6.49 raw-strip (sampled 8,000 articles) |
| **Total chunks** | **~30M** prose-clean (recommended) · ~45M raw |
| Ollama embed throughput | 32 chunks/s (nomic-768) · `st` reaches 57/s same model, 152/s at bge-small-384 |

### Time to embed everything — MEASURED (GTX 1080, batch 64)

| Backend / model | Dim | Pure-embed | End-to-end* | ~30M chunks | Quality | Disk |
|---|---|---|---|---|---|---|
| Ollama `nomic-embed-text` | 768 | 32/s | — | ~11 days | strong | 150 GB |
| **`st` `nomic-embed-text-v1.5`** | 768 | 57/s | **~43/s** | **~8 days** | strong | 150 GB |
| `st` `bge-small-en-v1.5` | 384 | 152/s | ~110/s | ~3 days | near-nomic | ~90 GB |
| `st` `all-MiniLM-L6-v2` | 384 | 286/s | ~190/s | ~1.8 days | weaker (dated) | ~90 GB |

\* End-to-end is the real serial-pipeline rate (read+clean+chunk+embed+COPY),
measured; it's ~15-25% under the pure-embed number because the GPU waits on the
CPU read between batches. Pure-embed came from `bench_embed.py`.

The job is **GPU-compute-bound**: pure-embed throughput scales inversely with
model size (137M→57/s, 33M→152/s, 22M→286/s), and Pascal has no usable fp16, so on
this card **model choice is the only real speed lever** — batching and a faster
server (TEI, which anyway needs Turing/SM7.5+ the 1080 lacks) don't help. A
threaded reader doesn't help either: it's already 4-20× faster than embed, and a
CPU-bound reader thread just contends the GIL and starves the GPU.

**Chosen: `nomic-embed-text-v1.5` (768d)** for max quality, ~8 days. Switch
`EMBED_MODEL`/`EMBED_DIM` in `.env` to trade quality for `bge-small` (~3 days).

Then a **separate** index build (`build_index.py`) — CPU/RAM-bound on the
Threadripper, **hours** for HNSW at 30M rows (much less for `ivfflat`).

### Disk in Postgres (~30M chunks)

| Config | Table | HNSW index | Total |
|---|---|---|---|
| **`halfvec` 768d** (default) | ~82 GB | ~69 GB | **~150 GB** |
| `vector` fp32 768d | ~129 GB | ~138 GB | ~270 GB |

Put it on the **empty `/mnt/nvme3`** (1.7 TB free). QLC is fine (see below).

## Run it

```bash
# 0. deps
pip install -r requirements.txt          # + a Pascal-capable CUDA torch (see requirements.txt)
cp .env.example .env ; set -a; . ./.env; set +a

# 1. tuned NVMe-backed Postgres
mkdir -p /mnt/nvme3/wiki-pgdata
docker compose -f docker-compose.wiki.yml up -d

# 2. how fast will this actually be? (turns the estimate into a number)
python bench_embed.py

# 3. validate quality on a subset FIRST
MAX_ARTICLES=50000 python ingest.py
python build_index.py
python query.py "what causes the northern lights?"

# 4. happy? run the full thing as a durable service (see below).
```

## Run the full job as a service (durable, resumable)

For a multi-day run, don't babysit a terminal — use the systemd unit. It resumes
from the DB cursor after a crash/OOM/reboot, and a clean `stop` frees the GPU.

```bash
sudo cp wiki-ingest.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wiki-ingest     # start + auto-start on boot

journalctl -u wiki-ingest -f                # live progress (%, ch/s, ETA)
sudo systemctl stop  wiki-ingest            # PAUSE (frees the GPU for other work)
sudo systemctl start wiki-ingest            # RESUME from the cursor

# when ingest finishes ([done] in the log), build the index once:
/mnt/nvme3/rag-venv/bin/python build_index.py
/mnt/nvme3/rag-venv/bin/python query.py "what causes the northern lights?"
```

Progress any time (bypasses logs, reads the cursor):
```sql
SELECT next_entry_id, chunks_done,
       round(100.0*next_entry_id/19707079,2) AS pct FROM ingest_cursor;
```

## Querying — `query.py` (one-shot) vs `chat.py` (agentic)

Both embed the question locally (`nomic-embed-text`, `search_query:` prefix) and
retrieve via pgvector cosine; generation is a local Ollama model. **Source the
`.env` first** — `config.py`'s default `db_url` points at the old `:5462/rag`,
the real DB is `:5463/wiki`.

```bash
set -a; . ./.env; set +a

# one-shot RAG: embed → top-k → grounded, cited answer
python query.py "what causes the northern lights?"

# agentic: multi-query planning + refine loop + multi-turn + [n] citations
python chat.py                        # interactive REPL (':q' to quit)
python chat.py "compare the causes of WWI and WWII"   # one-shot
```

`chat.py` adds, over `query.py`:
- **multi-query planning** — the model rewrites your question into several
  diverse search queries, so one bad phrasing doesn't sink recall;
- **agentic refine loop** — after seeing the first evidence it may issue
  follow-up searches (multi-hop) before answering;
- **multi-turn memory** in the REPL; each turn prints its searches live.

Tuning (env vars):

| var | default | effect |
|---|---|---|
| `AGENT_MODEL` | `llama3.1:8b` | generator; `qwen2.5:14b` is sharper but spills to CPU while the ingest holds the GPU |
| `AGENT_PLAN_MAX` | `3` | max search queries per question |
| `AGENT_REFINE_STEPS` | `2` | extra follow-up search rounds |
| `AGENT_CTX_CHUNKS` | `10` | passages shown to the model |

> **Speed:** until `build_index.py` has built the HNSW index, every search is an
> exact seqscan over the whole table (~21 s on 15M+ rows, measured). After
> indexing it drops to milliseconds. `chat.py` runs several searches per turn, so
> it is really meant for the post-index index.

## Serving it on the web — `ask_web.py`

A small read-only front end (Flask + waitress) exposing two endpoints and a
single-page UI. `pip install flask waitress`, then:

```bash
set -a; . ./.env; set +a
waitress-serve --listen=127.0.0.1:3600 --threads=8 ask_web:app
```

| endpoint | cost | notes |
|---|---|---|
| `GET /api/search?q=&k=` | ~0.5 s, no GPU | pure pgvector similarity, one chunk per article |
| `GET /api/ask?q=` | ~8 s, holds the GPU | retrieve → generate, `[n]` citations |
| `GET /healthz` | — | today's answer count vs the daily budget |

**The two are deliberately asymmetric, and that asymmetry is the whole
protection design.** A requests-per-second limit is the wrong control for
`/api/ask`: the binding resource is not request rate but a single GPU serving
one generation at a time. Ten clients each politely sending 1 req/s still queue
forever. So put the crude flood limits in your reverse proxy (`limit_req`,
`limit_conn`, body size) and keep the expensive-resource limits here:

- `BoundedSemaphore(1)` around generation, with a **bounded wait** — return 503
  rather than let a queue build behind the GPU;
- per-IP sliding windows for answers and searches;
- a **global daily answer budget** — denial-of-wallet control, with the GPU as
  the wallet;
- a question-length cap.

All in-process memory, because one GPU means one worker; counters reset on
restart, which is fine — they bound abuse, they don't bill anyone. Tune with
`ASK_DAILY_BUDGET`, `ASK_IP_HOURLY`, `ASK_IP_SEARCH_HOURLY`, `ASK_GEN_WAIT`,
`ASK_GEN_TIMEOUT`, `ASK_MAX_Q`.

> Testing rate limits **sequentially proves nothing** — six sequential 8-second
> requests stay inside any sane token bucket and all return 200. Test with real
> concurrency: `seq 1 40 | xargs -P 40 -I{} curl -s -o /dev/null -w "%{http_code} " ...`

Content served this way is Wikipedia text: keep the CC BY-SA attribution and the
citation links that the UI already renders.

## How resume works

`ingest_cursor` is a one-row table holding `next_entry_id`. Each batch does the
COPY **and** the cursor bump in the *same transaction*, so a crash rolls back
both — no half-written batch, no duplicates. Restart = read cursor, continue.
Progress prints as `[42.10%] entry=... chunks=...`.

## Why these choices

- **Trim appendix sections** (References / External links / …) → ~35% fewer
  chunks, little answerable-content lost.
- **Overlapping, sentence-aware chunks** → facts spanning a boundary survive.
- **nomic task prefixes** (`search_document:` / `search_query:`) — the demo's
  `embed.js` omits them; that quietly hurts retrieval. Fixed here.
- **Load first, index once.** Inserting 30M rows into a live HNSW index is
  pathologically slow.
- **QLC NVMe is fine.** The whole build writes <1 TB once (≈0.2% of the drives'
  ~500 TBW life), and with 251 GB RAM the hot index largely lives in page cache.
  Just keep the data dir on NVMe, not the HDD.
