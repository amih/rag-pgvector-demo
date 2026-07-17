"""Gem hunter — raw semantic probes over the Wikipedia index, no LLM in the loop.

  python gems.py                       # run the built-in probe list
  python gems.py "a machine that was obsolete before it was finished"
  python gems.py --probes probes.txt --k 150 --skip 15 --out gems.md

Why this exists (and why it is not query.py):

  query.py / chat.py put an LLM on top of retrieval. That layer's job is to be
  *reasonable*: it summarises, normalises, and cites the safest passage. That is
  exactly wrong for exploration, where the interesting hit is the unreasonable
  one. So this prints the retrieved chunks and nothing else.

Three choices that matter for finding the obscure:

  --skip N        The nearest hit is the most *central* article for a phrasing,
                  i.e. the one you could have found on your own. Surprises live
                  in the mid-ranks. Skipping the first N drops the obvious.
  dedupe          One long article otherwise floods the whole result set with
                  its own chunks. We keep only each article's best chunk.
  drop hub pages  "List of ...", "Index of ...", "Outline of ..." are curated by
                  hand. If a list already exists, vector search cannot beat it —
                  so they are noise here. --keep-lists to disable.
"""
import argparse
import re
import sys

from config import cfg
from db import connect, vec_literal

# Probe archetypes: each asks for something whose answer is a *sentence inside an
# article about something else* — no category, no hub page, no obvious keyword.
DEFAULT_PROBES = [
    # -- nameless concepts: you know the shape, not the term --
    "a machine that was already obsolete before it was finished being built",
    "a building designed for a purpose it never actually served",
    "an English word that entered the language through a mistranslation or misreading",
    "a project abandoned when the money ran out, left half-built and still standing",

    # -- Wikipedia's own hedges: uncertainty has no category --
    "the reason for this has never been satisfactorily explained",
    "no photograph or portrait of him is known to exist",
    "its current whereabouts are unknown and it has not been seen since",
    "the last person who knew how to make it died without passing on the technique",
    "the only known surviving example of its kind",

    # -- structural rhymes: the pattern cuts across topics, so categories can't hold it --
    "an institution that outlived the purpose it was created for by centuries",
    "a technical standard kept in place only because fixing the error would break compatibility",
    "a tradition invented recently but widely believed to be ancient",
    "a rule or law that produced the exact opposite of its intended effect",
    "a measurement later found to be based on a mistake but kept anyway",

    # -- negative space: defined by what is missing --
    "a war that was never formally ended, only paused",
    "a recording or film that no longer exists and was never copied",
    "an invention whose true inventor is disputed or unknown despite being recent",

    # -- buried biography clauses --
    "a person who survived two separate unrelated disasters",
    "someone famous for one thing who spent their life trying to be known for another",
    "an amateur whose side hobby outlasted their professional work",
]


def build_backend(kind):
    """Query-embedding backend.

    Default 'ollama': the ingest owns the GPU with its own sentence-transformers
    model, and cosine distance is scale-invariant, so Ollama's nomic-embed-text
    ranks the same as the ST model used at ingest time — without a second copy
    of the weights competing for the 1080's 8 GB.
    """
    import embed_backends
    if kind == "st":
        return embed_backends.STBackend()
    cfg.embed_model = "nomic-embed-text"   # Ollama's tag, not the HF repo id
    return embed_backends.OllamaBackend()


HUB_RE = re.compile(r"^(list|index|outline|glossary|timeline|comparison) of ", re.I)


def snippet(text, width):
    return re.sub(r"\s+", " ", text).strip()[:width]


def search(conn, qvec, k, skip, pool_mult, keep_lists):
    """Nearest chunks, deduped to one row per article, hub pages dropped.

    Filtering happens in Python rather than SQL WHERE: pgvector applies a WHERE
    *after* the HNSW scan, so a filtered query silently returns fewer rows than
    asked. Over-fetch an unfiltered pool and trim here instead.
    """
    want = k + skip
    pool = want * pool_mult
    lit = vec_literal(qvec)
    with conn.cursor() as cur:
        # ef_search must exceed the number of rows we want or recall collapses.
        # set_config(..., true) == SET LOCAL, but takes a bound parameter; plain
        # SET LOCAL only accepts a literal.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(max(cfg.ef_search, pool)),))
        cur.execute(
            f"""SELECT article_id, title, content,
                       embedding <=> %s::{cfg.vector_type} AS dist
                  FROM wiki_chunks
                 ORDER BY embedding <=> %s::{cfg.vector_type}
                 LIMIT %s""",
            (lit, lit, pool),
        )
        rows = cur.fetchall()

    best = {}
    for article_id, title, content, dist in rows:
        if not keep_lists and HUB_RE.match(title or ""):
            continue
        cur_best = best.get(article_id)
        if cur_best is None or dist < cur_best[2]:
            best[article_id] = (title, content, dist)
    hits = sorted(best.values(), key=lambda r: r[2])
    return hits[skip: skip + k], len(rows)


def main():
    ap = argparse.ArgumentParser(description="Raw semantic probes, no LLM.")
    ap.add_argument("query", nargs="*", help="one-off probe (default: built-in list)")
    ap.add_argument("--probes", help="file with one probe per line (# comments ok)")
    ap.add_argument("-k", type=int, default=40, help="hits to show per probe (default 40)")
    ap.add_argument("--skip", type=int, default=0,
                    help="drop the N nearest hits — the obvious ones (try 10-20)")
    ap.add_argument("--pool-mult", type=int, default=4,
                    help="over-fetch factor before dedupe/filter (default 4)")
    ap.add_argument("--width", type=int, default=260, help="snippet chars (default 260)")
    ap.add_argument("--keep-lists", action="store_true", help="keep 'List of ...' hub pages")
    ap.add_argument("--embed", choices=("ollama", "st"), default="ollama",
                    help="query-embedding backend (default ollama: leaves the GPU alone)")
    ap.add_argument("--out", help="also write markdown here")
    args = ap.parse_args()

    if args.query:
        probes = [" ".join(args.query)]
    elif args.probes:
        with open(args.probes, encoding="utf-8") as fh:
            probes = [l.strip() for l in fh
                      if l.strip() and not l.lstrip().startswith("#")]
    else:
        probes = DEFAULT_PROBES

    backend = build_backend(args.embed)
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    with connect() as conn:
        cur = conn.cursor()
        # Both cheap: one row from the cursor table, one PK-ordered row. Avoid
        # count(*)/max(title) here — they are seqscans over 20M+ rows.
        cur.execute("SELECT chunks_done FROM ingest_cursor WHERE id = 1")
        (chunks_done,) = cur.fetchone() or (0,)
        # Ingest is serial in ZIM order, which is alphabetical, so the newest row
        # is the alphabetical high-water mark.
        cur.execute("SELECT title FROM wiki_chunks ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        hwm = row[0] if row else ""

        emit(f"# Gem hunt — {len(probes)} probe(s)")
        emit()
        emit(f"Corpus: {chunks_done:,} chunks ingested. "
             f"Alphabetical high-water mark: `{hwm[:48]}`")
        if hwm[:1].upper() < "M":
            emit()
            emit("> **Partial corpus.** The ingest walks the ZIM alphabetically and has "
                 "not passed 'M' yet, so roughly half of Wikipedia (M-Z) is missing. "
                 "Thematic probes are silently blind to it — results will backfill with "
                 "weaker A-L matches. Re-run once the ingest reports `[done]`.")
        emit()

        for probe in probes:
            qvec = backend.embed_query(probe)
            hits, npool = search(conn, qvec, args.k, args.skip,
                                 args.pool_mult, args.keep_lists)
            emit(f"## {probe}")
            emit()
            if not hits:
                emit("_no hits_")
                emit()
                continue
            for i, (title, content, dist) in enumerate(hits, start=args.skip + 1):
                emit(f"**{i}.** `{dist:.3f}`  **{title}**")
                emit(f"    {snippet(content, args.width)}…")
            emit()
            sys.stderr.write(f"[probe] {probe[:56]:56} -> {len(hits)} hits "
                             f"(pool {npool})\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        sys.stderr.write(f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
