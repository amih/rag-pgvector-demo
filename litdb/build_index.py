"""Build the HNSW index on pubmed_emb (run once embedding is done; hours, CPU).
Rows added later go into the index automatically."""
from common import connect

with connect(autocommit=True) as conn:
    conn.execute("SET maintenance_work_mem = '16GB'")
    conn.execute("SET max_parallel_maintenance_workers = 8")
    conn.execute("CREATE INDEX IF NOT EXISTS pubmed_emb_hnsw ON pubmed_emb "
                 "USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)")
    conn.execute("ANALYZE pubmed_emb")
print("index ready")
