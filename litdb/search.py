"""Semantic search over PubMed abstracts.

  python search.py "natural tubulin polymerization inhibitors in food" [-k 15]
        [--from 2010] [--mesh "Anthelmintics"] [--prio1]

Exact scan until build_index.py has run (then HNSW, milliseconds).
"""
import argparse

from common import QUERY_PREFIX, connect, embed, vec_literal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=15)
    ap.add_argument("--from", dest="year_from", type=int)
    ap.add_argument("--mesh", help="require this MeSH descriptor (exact, case-sensitive)")
    ap.add_argument("--prio1", action="store_true", help="only the natural-products subset")
    a = ap.parse_args()
    q = vec_literal(embed([QUERY_PREFIX + a.query], keep_alive="5m")[0])
    where, params = [], {"q": q, "k": a.k}
    if a.year_from:
        where.append("a.year >= %(y)s"); params["y"] = a.year_from
    if a.mesh:
        where.append("%(m)s = ANY(a.mesh)"); params["m"] = a.mesh
    if a.prio1:
        where.append("a.priority = 1")
    filt = ("WHERE " + " AND ".join(where)) if where else ""
    # Filters are applied after a widened ANN candidate set when the index exists.
    sql = f"""
        WITH cand AS (SELECT pmid, embedding <=> %(q)s::halfvec AS d FROM pubmed_emb
                      ORDER BY embedding <=> %(q)s::halfvec LIMIT {a.k * 40})
        SELECT a.pmid, a.year, a.journal, a.title, a.doi, left(a.abstract, 300), c.d
        FROM cand c JOIN pubmed_articles a USING (pmid) {filt}
        ORDER BY c.d LIMIT %(k)s"""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 400")
        cur.execute(sql, params)
        for pmid, year, journal, title, doi, snip, d in cur.fetchall():
            print(f"\n[{1 - d:.3f}] {title}\n  {year} · {journal} · PMID {pmid}"
                  f"{' · doi:' + doi if doi else ''}\n  https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                  f"\n  {(snip or '').replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
