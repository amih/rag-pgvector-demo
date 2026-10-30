"""Embed PubMed title+abstract into pubmed_emb, yielding the GPU to other jobs.

Order: priority 1 (natural products / parasites / tubulin, ~10%) first, then the
rest. Keyset pass over pubmed_articles skipping rows already embedded; when a
pass finds nothing it sleeps and starts again, so it keeps up with the loader
and with daily updates. Kill-safe: each batch commits on its own.
"""
import os
import signal
import time

from common import DOC_PREFIX, connect, embed, vec_literal
from gpu_gate import busy

BATCH = int(os.environ.get("EMBED_BATCH", 64))
MAX_CHARS = int(os.environ.get("MAX_CHARS", 6000))
IDLE_SLEEP = int(os.environ.get("IDLE_SLEEP", 600))
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
    done, t0 = 0, time.time()
    while not STOP:
        did_any = False
        for prio in (1, 2):
            after = 0
            while not STOP:
                with conn.cursor() as cur:
                    rows = batch(cur, prio, after)
                conn.commit()
                if not rows:
                    break
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
                after, did_any = rows[-1][0], True
                done += len(rows)
                if done % (BATCH * 50) < BATCH:
                    rate = done / (time.time() - t0)
                    print(f"[p{prio}] pmid>{after:,} +{done:,} this run, {rate:.1f}/s wall",
                          flush=True)
        if not did_any and not STOP:
            time.sleep(IDLE_SLEEP)   # caught up; wait for the loader / updates
    conn.close()
    print(f"[stop] embedded {done:,} this run", flush=True)


if __name__ == "__main__":
    main()
