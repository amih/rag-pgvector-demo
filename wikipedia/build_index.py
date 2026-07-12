"""Build the ANN index AFTER the bulk load (never during it).

Inserting 30M+ rows into an existing HNSW index is pathologically slow; loading
first and indexing once is far faster. This is CPU/RAM-bound (your Threadripper +
251GB RAM), not GPU. Crank maintenance_work_mem and parallel workers via .env.

HNSW  = best recall/latency, slow heavy build.
IVFFlat = much faster/cheaper build, slightly lower recall — good for a first pass.
"""
import time
from config import cfg
from db import connect


def main():
    conn = connect()
    conn.autocommit = True
    opclass = f"{cfg.vector_type}_cosine_ops"
    with conn.cursor() as cur:
        cur.execute(f"SET maintenance_work_mem = '{cfg.maintenance_work_mem}'")
        cur.execute(f"SET max_parallel_maintenance_workers = {cfg.parallel_workers}")
        cur.execute(f"SET max_parallel_workers = {cfg.parallel_workers + 4}")
        cur.execute("SELECT count(*) FROM wiki_chunks")
        n = cur.fetchone()[0]
        print(f"indexing {n:,} rows with {cfg.index_type} ({opclass}) ...", flush=True)

        t = time.time()
        if cfg.index_type == "hnsw":
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS wiki_chunks_emb_hnsw "
                f"ON wiki_chunks USING hnsw (embedding {opclass})")
        else:
            lists = max(100, int(n ** 0.5))
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS wiki_chunks_emb_ivf "
                f"ON wiki_chunks USING ivfflat (embedding {opclass}) WITH (lists={lists})")
        print(f"index built in {(time.time() - t) / 60:.1f} min; running ANALYZE ...", flush=True)
        cur.execute("ANALYZE wiki_chunks")
    conn.close()
    print("done.")


if __name__ == "__main__":
    main()
