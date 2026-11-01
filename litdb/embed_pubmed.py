"""Embed PubMed title+abstract into pubmed_emb, yielding the GPU to other jobs.

Order: priority 1 (natural products / parasites / tubulin, ~10%) first, then the
rest, returning to priority 1 every RECHECK batches. Keyset pass over
pubmed_articles skipping rows already embedded; when a pass finds nothing it
sleeps and starts again, so it keeps up with the loader and with daily updates. Kill-safe: each batch commits on its own.
"""
import os
import signal
import time

from common import DOC_PREFIX, connect, embed, vec_literal
from gpu_gate import busy

BATCH = int(os.environ.get("EMBED_BATCH", 64))
MAX_CHARS = int(os.environ.get("MAX_CHARS", 6000))
IDLE_SLEEP = int(os.environ.get("IDLE_SLEEP", 600))
RECHECK = int(os.environ.get("RECHECK_BATCHES", 500))
STOP = False


def _stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def wait_for_gpu():
    reason = busy()
    if not reason:
        return
    since = time.time()
    print(f"[yield] {reason}", flush=True)
    while reason and not STOP:
        time.sleep(30)
        reason = busy()
    print(f"[resume] after {(time.time() - since) / 60:.0f} min", flush=True)


def batch(cur, prio, after):
    cur.execute("""
        SELECT a.pmid, a.title, a.abstract FROM pubmed_articles a
        WHERE a.priority = %s AND a.pmid > %s AND a.abstract IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM pubmed_emb e WHERE e.pmid = a.pmid)
        ORDER BY a.pmid LIMIT %s""", (prio, after, BATCH))
    return cur.fetchall()


def main():
    conn = connect()
    done, t0, since_check = 0, time.time(), 0
    prio, after, did_any = 1, {1: 0, 2: 0}, False
    while not STOP:
        with conn.cursor() as cur:
            rows = batch(cur, prio, after[prio])
        conn.commit()
        if not rows:
            if prio == 1:
                prio, after[1] = 2, 0
            elif did_any:              # finished a pass: look again from the start
                prio, after[2], did_any = 1, 0, False
            else:                      # caught up; wait for the loader / updates
                after[2] = 0
                prio = 1
                time.sleep(IDLE_SLEEP)
            continue
        wait_for_gpu()
        if STOP:
            break
        texts = [DOC_PREFIX + (t + "\n" + a)[:MAX_CHARS] for _, t, a in rows]
        embs = embed(texts)
        with conn.cursor() as cur:
            with cur.copy("COPY pubmed_emb (pmid, embedding) FROM STDIN") as cp:
                for (pmid, _, _), e in zip(rows, embs):
                    cp.write_row((pmid, vec_literal(e)))
        conn.commit()
        after[prio], did_any = rows[-1][0], True
        done += len(rows)
        if done % (BATCH * 50) < BATCH:
            rate = done / (time.time() - t0)
            print(f"[p{prio}] pmid>{after[prio]:,} +{done:,} this run, {rate:.1f}/s wall", flush=True)
        # The loader may still be adding priority 1 rows: go back for them regularly.
        if prio == 2:
            since_check += 1
            if since_check >= RECHECK:
                prio, after[1], since_check = 1, 0, 0
    conn.close()
    print(f"[stop] embedded {done:,} this run", flush=True)


if __name__ == "__main__":
    main()
