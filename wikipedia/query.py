"""Ask a question against the Wikipedia index.

  python query.py "why is the sky blue?"

Embeds the question (search_query: prefix), pulls the nearest chunks via pgvector,
and has Ollama answer strictly from them.
"""
import sys
import json
import urllib.request
from config import cfg
from db import connect, vec_literal
from embed_backends import get_backend


def generate(question, passages):
    context = "\n\n".join(f"[{i+1}] {t}: {c}" for i, (t, c, _) in enumerate(passages))
    prompt = (
        "Answer the question using ONLY the context below. Cite sources as [n]. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    body = json.dumps({
        "model": cfg.chat_model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": cfg.num_ctx, "temperature": cfg.temperature},
    }).encode()
    req = urllib.request.Request(cfg.ollama_url.rstrip("/") + "/api/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]


def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('usage: python query.py "your question"')
        return

    qvec = vec_literal(get_backend().embed_query(q))
    conn = connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {cfg.ef_search}")
        cur.execute(
            f"SELECT title, content, 1 - (embedding <=> %s::{cfg.vector_type}) AS sim "
            f"FROM wiki_chunks ORDER BY embedding <=> %s::{cfg.vector_type} LIMIT %s",
            (qvec, qvec, cfg.top_k))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No rows yet — run ingest.py + build_index.py first.")
        return
    print("\nRetrieved:")
    for t, c, s in rows:
        print(f"  [{s:.3f}] {t}")
    print("\nAnswer:\n" + generate(q, rows))


if __name__ == "__main__":
    main()
