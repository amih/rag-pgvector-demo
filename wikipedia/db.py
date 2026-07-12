"""Postgres helpers (psycopg 3). Schema is created from `cfg` so the vector
type/dim are a single source of truth."""
import psycopg
from config import cfg


def connect():
    return psycopg.connect(cfg.db_url, autocommit=False)


def vec_literal(vec):
    # pgvector's text input format, valid for both `vector` and `halfvec`.
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS wiki_chunks (
                id          bigserial PRIMARY KEY,
                article_id  int  NOT NULL,
                chunk_index int  NOT NULL,
                title       text NOT NULL,
                content     text NOT NULL,
                embedding   {cfg.vector_type}({cfg.embed_dim}) NOT NULL
            )
        """)
        # Single-row cursor = the whole resume mechanism. Updated in the SAME
        # transaction as each COPY, so a crash rolls back both -> clean restart.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingest_cursor (
                id            int    PRIMARY KEY,
                next_entry_id int    NOT NULL DEFAULT 0,
                chunks_done   bigint NOT NULL DEFAULT 0
            )
        """)
        cur.execute("INSERT INTO ingest_cursor (id) VALUES (1) ON CONFLICT DO NOTHING")
    conn.commit()
