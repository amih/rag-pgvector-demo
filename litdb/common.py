"""Shared config + DB helpers for the PubMed index (litdb).

Same Postgres container as the Wikipedia index (wiki-rag, :5463), separate
database `litdb`. Embeddings: nomic-embed-text via Ollama (see gpu_gate.py for
why Ollama rather than in-process torch).
"""
import json
import os
import urllib.request

import psycopg

RAW       = os.environ.get("RAW", "/mnt/hdd/litdb")
DB_URL    = os.environ.get("LITDB_URL", "postgres://rag:rag@localhost:5463/litdb")
OLLAMA    = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMB_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def connect(autocommit=False):
    return psycopg.connect(DB_URL, autocommit=autocommit)


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def embed(texts, keep_alive="1m"):
    # Short keep_alive: when we pause for another GPU job, nomic leaves VRAM fast.
    body = json.dumps({"model": EMB_MODEL, "input": texts, "keep_alive": keep_alive,
                       "truncate": True, "options": {"num_ctx": 2048}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["embeddings"]


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS pubmed_articles (
    pmid      bigint PRIMARY KEY,
    title     text NOT NULL,
    abstract  text,
    journal   text,
    year      smallint,
    doi       text,
    pmcid     text,
    mesh      text[],
    chemicals text[],
    pubtypes  text[],
    priority  smallint NOT NULL       -- 1 = natural products / parasites / tubulin; 2 = rest
);
CREATE INDEX IF NOT EXISTS pubmed_articles_prio ON pubmed_articles (priority, pmid);
CREATE TABLE IF NOT EXISTS pubmed_emb (
    pmid      bigint PRIMARY KEY,
    embedding halfvec(768) NOT NULL
);
CREATE TABLE IF NOT EXISTS loaded_files (
    name      text PRIMARY KEY,
    records   int NOT NULL,
    deleted   int NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now()
);
"""
