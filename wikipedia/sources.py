"""One question, five databases. Each source is searched on its own and keeps its
own top results, so a big source (35.8M Wikipedia passages) cannot crowd out a
small one (43k Dr. Duke records).

  wiki    Wikipedia passages        vector search (nomic-embed, HNSW)
  pubmed  PubMed abstracts          vector search (same model, same query vector)
  chembl  ChEMBL molecules + assays keyword search (litdb src_docs)
  duke    Dr. Duke plants+chemicals keyword search
  lotus   LOTUS natural products    keyword search

Vector search finds meaning; the three structured databases are lists of names
(compounds, plants, targets), where the exact word is what counts, so they get
Postgres full-text search instead. See litdb/load_structured.py.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import psycopg

from config import cfg
from db import connect, vec_literal
from embed_backends import get_backend

LITDB_URL = os.environ.get("LITDB_URL", "postgres://rag:rag@localhost:5463/litdb")

SOURCES = {
    "wiki":   {"label": "Wikipedia", "about": "35.8M passages of English Wikipedia"},
    "pubmed": {"label": "PubMed", "about": "29.6M abstracts of biomedical papers"},
    "chembl": {"label": "ChEMBL", "about": "Drug-like molecules and lab assays (EMBL-EBI)"},
    "duke":   {"label": "Dr. Duke", "about": "Plants, their chemicals and folk uses (USDA)"},
    "lotus":  {"label": "LOTUS", "about": "Natural products and the organisms they come from"},
}
KEYWORD_SOURCES = ("chembl", "duke", "lotus")
KIND_LABEL = {"molecule": "molecule", "assay": "assay", "chemical": "chemical",
              "plant": "plant", "compound": "compound", "organism": "organism"}

# Words in more documents than this are not searched on in that source ("cell" or
# "activity" in ChEMBL's 2M assays): too many rows to score, and they say little.
MAX_TERM_DOCS = int(os.environ.get("ASK_MAX_TERM_DOCS", 100_000))


# The query vector: one short text, ~50 ms on the CPU with the same nomic model
# the documents were embedded with (identical vectors, cosine 0.99999 vs Ollama).
# Doing it on the CPU keeps search working while the GPU is busy with someone
# else's job. Set EMBED_BACKEND=st DEVICE=cpu for that (the service drop-in does).
_backend, _backend_lock = None, threading.Lock()


def embed_query(text):
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = get_backend()
    return _backend.embed_query(text)


def warm_up():
    threading.Thread(target=lambda: embed_query("warm up"), daemon=True).start()


def _wiki(qvec, k):
    conn = connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {max(cfg.ef_search, k * 8)}")
            cur.execute(
                f"SELECT title, content, 1 - (embedding <=> %s::{cfg.vector_type}) "
                f"FROM wiki_chunks ORDER BY embedding <=> %s::{cfg.vector_type} LIMIT %s",
                (qvec, qvec, k * 3))
            rows = cur.fetchall()
    finally:
        conn.close()
    seen, out = set(), []   # one passage per article
    for title, content, sim in rows:
        if title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "text": content, "score": round(float(sim), 3),
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "meta": ""})
    return out[:k]


def _pubmed(qvec, k, year_from=None):
    filt, params = "", {"q": qvec, "k": k}
    if year_from:
        filt, params["y"] = "WHERE a.year >= %(y)s", year_from
    sql = f"""
        WITH cand AS (SELECT pmid, embedding <=> %(q)s::halfvec AS d FROM pubmed_emb
                      ORDER BY embedding <=> %(q)s::halfvec LIMIT {k * (40 if year_from else 4)})
        SELECT a.pmid, a.year, a.journal, a.title, a.abstract, c.d
        FROM cand c JOIN pubmed_articles a USING (pmid) {filt}
        ORDER BY c.d LIMIT %(k)s"""
    with psycopg.connect(LITDB_URL, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 200")
        cur.execute(sql, params)
        return [{"title": title, "text": abstract or "", "score": round(1 - float(d), 3),
                 "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                 "meta": " · ".join(str(x) for x in (year, journal, f"PMID {pmid}") if x)}
                for pmid, year, journal, title, abstract, d in cur.fetchall()]


def _keyword(source, q, k):
    """Full-text search, any word may match, scored by how rare the matched words
    are across all three keyword databases (IDF). A rare name such as "allicin"
    outweighs a common word such as "activity" (frequent in ChEMBL), so a record
    that matches the name beats one that only matches the filler, and one that
    matches several rare words beats both. Words the source has never seen, or
    sees too often to be useful, are skipped."""
    with psycopg.connect(LITDB_URL, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SET statement_timeout = '5s'")
        cur.execute("""
            WITH l AS (SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(q)s))),
                 n AS (SELECT count(*)::float8 AS n FROM src_docs)
            SELECT l.lexeme, ln(n.n / sum(a.ndoc)) AS idf
            FROM l JOIN src_lexemes a ON a.lexeme = l.lexeme, n
            GROUP BY l.lexeme, n.n
            HAVING bool_or(a.source = %(s)s)
               AND max(a.ndoc) FILTER (WHERE a.source = %(s)s) <= %(max)s""",
                    {"q": q, "s": source, "max": MAX_TERM_DOCS})
        terms = cur.fetchall()
        if not terms:
            return []
        lex = [t for t, _ in terms]
        weight = [float(w) for _, w in terms]
        tsq = " | ".join("'" + t.replace("'", "''") + "'" for t in lex)
        cur.execute("""
            WITH t AS (SELECT * FROM unnest(%(lex)s::text[], %(w)s::float8[]) AS t(lex, w)),
            q AS (SELECT to_tsquery('simple', %(tsq)s) AS q),
            hits AS (
                SELECT d.kind, d.ext_id, d.title, d.body, d.url, m.score, m.matched
                FROM src_docs d, q,
                LATERAL (SELECT sum(t.w) + ts_rank_cd(d.tsv, q.q, 1) AS score,
                                array_agg(t.lex ORDER BY t.w DESC) AS matched
                         FROM t WHERE d.tsv @@ to_tsquery('simple', quote_literal(t.lex))) m
                WHERE d.source = %(s)s AND d.tsv @@ q.q
                ORDER BY m.score DESC LIMIT %(k)s)
            SELECT kind, ext_id, title, url, score, matched,
                   ts_headline('english', body, q.q,
                               'MaxFragments=3, MaxWords=30, MinWords=12, '
                               'StartSel="", StopSel="", FragmentDelimiter=" … "')
            FROM hits, q ORDER BY score DESC""",
                    {"lex": lex, "w": weight, "tsq": tsq, "s": source, "k": k * 4})
        rows, seen = [], set()   # ChEMBL repeats one assay text for many dose levels
        for r in cur.fetchall():
            if r[2] not in seen:
                seen.add(r[2])
                rows.append(r)
        return [{"title": title, "text": text, "score": round(float(score), 2),
                 "url": url, "meta": f"{KIND_LABEL.get(kind, kind)} · {ext_id}",
                 "matched": matched}
                for kind, ext_id, title, url, score, matched, text in rows[:k]]


def search_sources(q, sources, k=5, year_from=None):
    """Returns {source: [hit, ...]} for every requested source (empty list = no
    match; {"error": ...} entries if that source failed)."""
    sources = [s for s in sources if s in SOURCES] or list(SOURCES)
    qvec = None
    if "wiki" in sources or "pubmed" in sources:
        qvec = vec_literal(embed_query(q))

    jobs = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        for s in sources:
            if s == "wiki":
                jobs[s] = pool.submit(_wiki, qvec, k)
            elif s == "pubmed":
                jobs[s] = pool.submit(_pubmed, qvec, k, year_from)
            else:
                jobs[s] = pool.submit(_keyword, s, q, k)
        out = {}
        for s, fut in jobs.items():
            try:
                out[s] = fut.result(timeout=30)
            except Exception as e:   # one broken source must not hide the others
                out[s] = {"error": type(e).__name__}
    return out
