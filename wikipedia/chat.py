"""Agentic chat over the Wikipedia index.

  python chat.py                                   # interactive REPL
  python chat.py "compare the causes of WWI and WWII"   # one-shot, then exit

How this differs from query.py (plain single-shot RAG):
  * multi-query planning  - the model rewrites your question into a few diverse
                            search queries, so one awkward phrasing doesn't sink
                            recall.
  * agentic refine loop   - after seeing the first evidence the model may issue
                            follow-up searches (multi-hop) before it answers.
  * multi-turn memory     - follow-ups resolve against the conversation so far.
  * inline [n] citations  - every claim maps to a retrieved passage; the sources
                            are printed under each answer.

Retrieval stays 100% local: nomic-embed-text (Ollama) -> pgvector cosine.
Generation is a local Ollama model. Default llama3.1:8b, which fits on the 8 GB
GTX 1080 *alongside* the running ingest; qwen2.5:14b is sharper but spills to CPU
while the ingest holds the GPU. Override with AGENT_MODEL=... .

NOTE: until build_index.py has built the HNSW index, each search is an exact
seqscan over the whole table (slow: seconds-to-minutes on 15 M+ rows). After
indexing it drops to milliseconds. We enable parallel workers to soften the
pre-index case; it is harmless once the index exists (HNSW won't seqscan).
"""
import os
import re
import sys
import json
import time
import urllib.request

from config import cfg
from db import connect, vec_literal

# --- knobs (env-overridable, sensible for the home box) ---
GEN_MODEL     = os.environ.get("AGENT_MODEL", "llama3.1:8b")
EMBED_MODEL   = os.environ.get("AGENT_EMBED_MODEL", "nomic-embed-text")  # Ollama tag
PLAN_MAX      = int(os.environ.get("AGENT_PLAN_MAX", 3))     # search queries per plan
REFINE_STEPS  = int(os.environ.get("AGENT_REFINE_STEPS", 2)) # extra search rounds
CTX_CHUNKS    = int(os.environ.get("AGENT_CTX_CHUNKS", 10))  # passages shown to model
HISTORY_TURNS = int(os.environ.get("AGENT_HISTORY_TURNS", 4))
PARALLEL      = int(os.environ.get("AGENT_EXACT_PARALLEL", 4))  # pre-index seqscan help


# ---------------------------------------------------------------- Ollama calls
def _ollama(path, payload, timeout=600):
    req = urllib.request.Request(
        cfg.ollama_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def embed_query(text):
    # nomic REQUIRES the search_query: prefix; cosine (<=>) is scale-invariant so
    # Ollama vs the ST ingest backend rank identically.
    out = _ollama("/api/embed", {"model": EMBED_MODEL,
                                 "input": cfg.query_prefix + text})
    return out["embeddings"][0]


def generate(prompt, temperature=None):
    out = _ollama("/api/generate", {
        "model": GEN_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_ctx": cfg.num_ctx,
                    "temperature": cfg.temperature if temperature is None else temperature},
    })
    return out["response"].strip()


# ------------------------------------------------------------------- retrieval
class Retriever:
    """Accumulates de-duplicated passages across every search in one turn and
    hands the model the top CTX_CHUNKS by similarity, with stable [n] ids."""

    def __init__(self, conn):
        self.conn = conn
        self.sources = []          # [{n, title, content, sim}]
        self.seen = set()          # (title, content[:120]) -> dedupe key

    def search(self, query, k=None):
        k = k or cfg.top_k
        qvec = vec_literal(embed_query(query))
        t0 = time.time()
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT title, content, 1 - (embedding <=> %s::{cfg.vector_type}) AS sim "
                f"FROM wiki_chunks ORDER BY embedding <=> %s::{cfg.vector_type} LIMIT %s",
                (qvec, qvec, k))
            rows = cur.fetchall()
        added = 0
        for title, content, sim in rows:
            key = (title, content[:120])
            if key in self.seen:
                continue
            self.seen.add(key)
            self.sources.append({"n": len(self.sources) + 1, "title": title,
                                 "content": content, "sim": float(sim)})
            added += 1
        print(f"    · search: {query!r}  ({added} new / {len(rows)} hits, "
              f"{time.time() - t0:.1f}s)", flush=True)
        return added

    def context(self):
        """Top CTX_CHUNKS passages by similarity, keeping their stable [n]."""
        top = sorted(self.sources, key=lambda s: s["sim"], reverse=True)[:CTX_CHUNKS]
        top = sorted(top, key=lambda s: s["n"])   # present in id order
        text = "\n\n".join(f"[{s['n']}] {s['title']}: {s['content']}" for s in top)
        return text, top


# ---------------------------------------------------------------- agent prompts
def _history_block(history):
    if not history:
        return ""
    turns = history[-HISTORY_TURNS * 2:]
    lines = "\n".join(f"{role.upper()}: {msg}" for role, msg in turns)
    return f"Conversation so far:\n{lines}\n\n"


def plan_queries(user_msg, history):
    prompt = (
        f"{_history_block(history)}"
        "You are planning a search over the full English Wikipedia to answer the "
        "user's latest message. Resolve any pronouns using the conversation.\n"
        f'User message: "{user_msg}"\n\n'
        f"Output ONLY a JSON array of up to {PLAN_MAX} short, diverse search "
        "queries (sub-questions or reworded phrasings) that together retrieve the "
        "facts needed. Example: [\"first query\", \"second query\"]")
    raw = generate(prompt, temperature=0.3)
    queries = []
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            queries = [q.strip() for q in json.loads(m.group(0))
                       if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            pass
    queries.append(user_msg)                     # always keep the raw question
    seen, out = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:PLAN_MAX + 1]


SEARCH_RE = re.compile(r"^\s*SEARCH:\s*(.+?)\s*$", re.M)


def answer_step(user_msg, history, evidence, force_answer):
    rules = (
        "You are a precise research assistant. Use ONLY the numbered evidence "
        "below. Cite every fact as [n] with the evidence numbers. If the evidence "
        "does not contain the answer, say you don't know.\n")
    if force_answer:
        rules += "Answer the question now from the evidence. Do NOT ask to search.\n"
    else:
        rules += (
            "If the evidence is INSUFFICIENT, do not answer yet: instead output "
            f"only search requests, one per line as `SEARCH: <query>` (max {PLAN_MAX}), "
            "and nothing else. If the evidence IS sufficient, write the final "
            "answer with [n] citations.\n")
    prompt = (
        f"{rules}\n{_history_block(history)}"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {user_msg}\nAnswer:")
    return generate(prompt)


# ------------------------------------------------------------------- one turn
def run_turn(retr, user_msg, history):
    for q in plan_queries(user_msg, history):
        retr.search(q)

    final, used = None, []
    for step in range(REFINE_STEPS + 1):
        force = step == REFINE_STEPS
        evidence, used = retr.context()
        out = answer_step(user_msg, history, evidence, force)
        wants = [] if force else SEARCH_RE.findall(out)
        if wants:
            added = sum(retr.search(q) for q in wants[:PLAN_MAX])
            if added == 0:            # nothing new to be found -> stop and answer
                evidence, used = retr.context()
                final = answer_step(user_msg, history, evidence, force_answer=True)
                break
            continue
        final = SEARCH_RE.sub("", out).strip()
        break

    print("\n" + final)
    if used:
        print("\nSources:")
        for s in used:
            print(f"  [{s['n']}] ({s['sim']:.3f}) {s['title']}")
    history += [("user", user_msg), ("assistant", final)]


# ----------------------------------------------------------------------- main
def open_conn():
    conn = connect()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {cfg.ef_search}")
        cur.execute(f"SET max_parallel_workers_per_gather = {PARALLEL}")
    return conn


def main():
    conn = open_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM wiki_chunks")
        n = cur.fetchone()[0]
        cur.execute("SELECT 1 FROM pg_indexes WHERE tablename='wiki_chunks' "
                    "AND indexdef ILIKE '%%hnsw%%'")
        indexed = cur.fetchone() is not None
    print(f"wiki-chat  |  gen={GEN_MODEL}  embed={EMBED_MODEL}  "
          f"chunks={n:,}  index={'hnsw' if indexed else 'NONE (exact seqscan, slow)'}")

    history = []
    if len(sys.argv) > 1:                     # one-shot
        run_turn(Retriever(conn), " ".join(sys.argv[1:]), history)
        return

    print("Ask a question. New search context per question. ':q' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in (":q", ":quit", "exit"):
            break
        if not q:
            continue
        try:
            run_turn(Retriever(conn), q, history)   # fresh evidence per question
        except Exception as e:                       # keep the REPL alive
            print(f"[error] {e}")
        print()


if __name__ == "__main__":
    main()
