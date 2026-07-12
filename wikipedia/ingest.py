"""Resumable Wikipedia -> pgvector ingester (serial, robust).

  ZIM entry ─▶ clean ─▶ chunk ─▶ embed (GPU) ─▶ COPY into Postgres + cursor bump (same txn)

Serial on purpose: the embed (~49 ch/s) is the bottleneck, while read+clean+chunk
(~200-970 ch/s) and format+COPY (<2% of a batch) are cheap, so overlapping them
buys ~15% — not worth a background thread that contends the GIL and starves the
GPU on a multi-day unattended run. (A multiprocessing reader could recover that
~15% later if wanted.)

Resume: kill it anytime (Ctrl-C, reboot, power cut). Re-run; it reads
`ingest_cursor` and continues from the last committed batch. The COPY and the
cursor bump commit together — no duplicates, no half-written batch.

Subset run: MAX_ARTICLES=50000 stops early; run again with no cap to continue.
"""
import signal
import time
from config import cfg
from db import connect, ensure_schema, vec_literal
from chunker import chunk
from zim_source import ZimSource
from embed_backends import get_backend

STOP = False


def _stop(*_):
    global STOP
    STOP = True
    print("\n[signal] finishing the current batch, then exiting cleanly...", flush=True)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def flush(conn, backend, buf):
    texts = [row[3] for row in buf]
    embs = []
    for i in range(0, len(texts), cfg.embed_batch):
        embs.extend(backend.embed_docs(texts[i:i + cfg.embed_batch]))
    last_eid = buf[-1][0]
    with conn.cursor() as cur:
        with cur.copy("COPY wiki_chunks (article_id, chunk_index, title, content, embedding) "
                      "FROM STDIN") as cp:
            for (aid, idx, title, content), emb in zip(buf, embs):
                cp.write_row((aid, idx, title, content, vec_literal(emb)))
        cur.execute(
            "UPDATE ingest_cursor SET next_entry_id=%s, chunks_done=chunks_done+%s WHERE id=1",
            (last_eid + 1, len(buf)))
    conn.commit()
    return len(buf)


def main():
    conn = connect()
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT next_entry_id, chunks_done FROM ingest_cursor WHERE id=1")
        start_id, done = cur.fetchone()

    backend = get_backend()
    src = ZimSource()
    print(f"[resume] entry_id={start_id:,}/{src.entry_count:,}  chunks_done={done:,}  "
          f"backend={cfg.embed_backend}  vec={cfg.vector_type}({cfg.embed_dim})", flush=True)

    buf, arts, new_arts, total = [], 0, 0, done
    t0 = time.time()
    for eid, title, text in src.iter_from(start_id):
        for cidx, c in enumerate(chunk(text, cfg.chunk_size, cfg.chunk_overlap)):
            buf.append((eid, cidx, title, c))
        arts += 1
        new_arts += 1
        if arts >= cfg.article_batch or STOP:
            if buf:
                n = flush(conn, backend, buf)
                total += n
                dt = time.time() - t0
                rate = (total - done) / dt if dt > 0 else 0
                eta = (30_000_000 - total) / rate / 86400 if rate > 0 else 0
                print(f"[{100 * eid / src.entry_count:5.2f}%] entry={eid:,} chunks={total:,} "
                      f"(+{n}) {rate:5.1f} ch/s  eta~{eta:.1f}d", flush=True)
            buf, arts = [], 0
            if STOP:
                break
            if cfg.max_articles and new_arts >= cfg.max_articles:
                print(f"[cap] MAX_ARTICLES={cfg.max_articles} reached; stopping.", flush=True)
                break

    if buf:
        total += flush(conn, backend, buf)
    conn.close()
    print(f"[{'paused' if STOP else 'done'}] total_chunks={total:,}. "
          f"Re-run to continue, then `python build_index.py`.", flush=True)


if __name__ == "__main__":
    main()
