"""Read-only web front end for the Wikipedia pgvector index.

Two endpoints, deliberately asymmetric, because they cost wildly different things:

  /api/search   pure pgvector similarity — no LLM. Milliseconds, no GPU. Cheap.
  /api/ask      retrieve + generate an answer with [n] citations. Seconds, and it
                occupies the whole GPU while it runs.

That asymmetry is the entire protection design. A request-per-second limit is the
wrong control for /api/ask: the binding resource is not request rate, it is the
single GPU, which serves exactly one generation at a time. Ten clients each
politely sending one request per second still queue forever. So:

  * nginx  enforces the crude flood limits (req/s, concurrent conns, body size).
  * here   enforces the expensive-resource limits — one generation at a time, a
           bounded wait for a slot, a per-IP hourly cap, and a global daily
           budget. Denial-of-wallet controls, with the GPU as the wallet.

Everything here is in-process memory: this is a single-worker service by design
(one GPU, one generation), so there is nothing to share. Counters reset on
restart, which is fine — they exist to bound abuse, not to bill anyone.

Run:  waitress-serve --listen=127.0.0.1:3600 ask_web:app
"""
import html
import json
import os
import threading
import time
import urllib.request
from collections import defaultdict, deque

from flask import Flask, jsonify, request

from config import cfg
from db import connect, vec_literal
from embed_backends import get_backend

app = Flask(__name__)

MAX_Q = int(os.environ.get("ASK_MAX_Q", 300))              # chars
GEN_SLOT_WAIT = float(os.environ.get("ASK_GEN_WAIT", 25))  # s to wait for the GPU
GEN_TIMEOUT = float(os.environ.get("ASK_GEN_TIMEOUT", 120))
IP_ASK_PER_HOUR = int(os.environ.get("ASK_IP_HOURLY", 20))
IP_SEARCH_PER_HOUR = int(os.environ.get("ASK_IP_SEARCH_HOURLY", 120))
DAILY_ASK_BUDGET = int(os.environ.get("ASK_DAILY_BUDGET", 300))

# One generation at a time. The 1080 has 8GB; llama3.1:8b fills it. A second
# concurrent generation does not run twice as slow, it thrashes.
_gpu = threading.BoundedSemaphore(1)
_lock = threading.Lock()
_hits = defaultdict(deque)          # (ip, kind) -> timestamps, trimmed to 1h
_day = {"stamp": None, "asks": 0}


def _client_ip() -> str:
    # nginx sets X-Forwarded-For and is the only thing that can reach us
    # (we bind 127.0.0.1), so the last hop is trustworthy here.
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "?")


def _take_slot(ip: str, kind: str, per_hour: int):
    """Sliding-window per-IP counter. Returns None if allowed, else an error."""
    now = time.time()
    with _lock:
        q = _hits[(ip, kind)]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= per_hour:
            return f"Rate limit: {per_hour} {kind} requests per hour. Try later."
        q.append(now)
    return None


def _take_daily_budget():
    """Global cap so one bad day cannot pin the GPU for everyone, including me."""
    today = time.strftime("%Y-%m-%d")
    with _lock:
        if _day["stamp"] != today:
            _day.update(stamp=today, asks=0)
        if _day["asks"] >= DAILY_ASK_BUDGET:
            return "Daily answer budget reached. Semantic search still works."
        _day["asks"] += 1
    return None


def _question():
    q = (request.args.get("q") or "").strip()
    if not q:
        return None, "Missing ?q="
    if len(q) > MAX_Q:
        return None, f"Question too long (max {MAX_Q} characters)."
    return q, None


def retrieve(q, k):
    qvec = vec_literal(get_backend().embed_query(q))
    conn = connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {cfg.ef_search}")
            cur.execute(
                f"SELECT title, content, 1 - (embedding <=> %s::{cfg.vector_type}) "
                f"FROM wiki_chunks ORDER BY embedding <=> %s::{cfg.vector_type} LIMIT %s",
                (qvec, qvec, k))
            return cur.fetchall()
    finally:
        conn.close()


def generate(question, passages):
    context = "\n\n".join(f"[{i+1}] {t}: {c}" for i, (t, c, _) in enumerate(passages))
    prompt = ("Answer the question using ONLY the context below. Cite sources as [n]. "
              "If the answer is not in the context, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    body = json.dumps({
        "model": cfg.chat_model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": cfg.num_ctx, "temperature": cfg.temperature},
    }).encode()
    req = urllib.request.Request(cfg.ollama_url.rstrip("/") + "/api/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
        return json.loads(r.read())["response"]


@app.get("/api/search")
def api_search():
    q, err = _question()
    if err:
        return jsonify(error=err), 400
    if (err := _take_slot(_client_ip(), "search", IP_SEARCH_PER_HOUR)):
        return jsonify(error=err), 429

    k = min(int(request.args.get("k", 8) or 8), 25)
    t0 = time.time()
    rows = retrieve(q, k)
    # One chunk per article: a long article otherwise floods its own result set.
    seen, out = set(), []
    for title, content, sim in rows:
        if title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "similarity": round(float(sim), 4),
                    "extract": content[:600],
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")})
    return jsonify(query=q, took_ms=round((time.time() - t0) * 1000, 1), results=out)


@app.get("/api/ask")
def api_ask():
    q, err = _question()
    if err:
        return jsonify(error=err), 400
    ip = _client_ip()
    if (err := _take_slot(ip, "ask", IP_ASK_PER_HOUR)):
        return jsonify(error=err), 429
    if (err := _take_daily_budget()):
        return jsonify(error=err), 429

    t0 = time.time()
    rows = retrieve(q, cfg.top_k)
    if not rows:
        return jsonify(error="No results."), 404

    # Bounded wait: fail fast rather than let a queue build behind the GPU.
    if not _gpu.acquire(timeout=GEN_SLOT_WAIT):
        return jsonify(error="Busy — one answer is generated at a time. "
                             "Try again shortly, or use /api/search."), 503
    try:
        answer = generate(q, rows)
    except Exception as e:
        return jsonify(error=f"Generation failed: {type(e).__name__}"), 502
    finally:
        _gpu.release()

    return jsonify(
        query=q, answer=answer, took_ms=round((time.time() - t0) * 1000, 1),
        sources=[{"n": i + 1, "title": t,
                  "url": "https://en.wikipedia.org/wiki/" + t.replace(" ", "_")}
                 for i, (t, _, _) in enumerate(rows)])


@app.get("/healthz")
def healthz():
    with _lock:
        return jsonify(ok=True, asks_today=_day["asks"], budget=DAILY_ASK_BUDGET)


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Ask Wikipedia — semantic search over 35.8M chunks</title>
<style>
:root{color-scheme:light dark;--fg:#1a1a1a;--bg:#fdfdfb;--muted:#5a5a5a;--line:#d8d5cc;--accent:#8a6a2f;--card:#f2f0ea}
@media(prefers-color-scheme:dark){:root{--fg:#e8e6e1;--bg:#16161a;--muted:#a3a09a;--line:#33323a;--accent:#d4a94a;--card:#20202a}}
*{box-sizing:border-box}
body{margin:0 auto;padding:2rem 1.25rem 5rem;max-width:44rem;background:var(--bg);color:var(--fg);
font:1.0625rem/1.65 Georgia,'Times New Roman',serif}
h1{font-size:1.6rem;margin:0 0 .25rem}
p.sub{color:var(--muted);font-size:.95rem;margin:0 0 1.5rem}
form{display:flex;gap:.5rem;flex-wrap:wrap}
input[type=text]{flex:1 1 16rem;padding:.7rem .8rem;font:inherit;font-size:1rem;
background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}
button{padding:.7rem 1.1rem;font:inherit;font-size:.95rem;cursor:pointer;border-radius:4px;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{border-color:var(--accent);color:var(--accent)}
.hint{color:var(--muted);font-size:.85rem;margin:.6rem 0 0}
#out{margin-top:2rem}
.answer{background:var(--card);padding:1rem 1.2rem;border-radius:4px;border-left:3px solid var(--accent);white-space:pre-wrap}
.hit{border-top:1px solid var(--line);padding:.9rem 0}
.hit a{color:var(--accent);text-decoration:none;font-weight:600}
.hit p{margin:.3rem 0 0;font-size:.95rem;color:var(--fg)}
.score{color:var(--muted);font-size:.8rem;font-family:ui-monospace,monospace}
.err{color:#b4451f}
footer{margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem;color:var(--muted);font-size:.85rem}
footer a{color:var(--accent)}
</style></head><body>
<h1>Ask Wikipedia</h1>
<p class=sub>Semantic search over the full English Wikipedia &mdash; 35.8 million passages,
768-dimension embeddings, HNSW index in Postgres. Self-hosted on one GPU at home.</p>
<form id=f>
  <input type=text id=q maxlength=__MAXQ__ placeholder="Ask a question, or describe what you're looking for" autofocus>
  <button type=submit class=primary id=bAsk>Answer</button>
  <button type=button id=bSearch>Search only</button>
</form>
<p class=hint><b>Search only</b> is pure vector similarity &mdash; no language model, instant.
<b>Answer</b> runs a local model over the retrieved passages; one at a time, so it may queue.</p>
<div id=out></div>
<footer>
Content from <a href="https://en.wikipedia.org">English Wikipedia</a>, licensed
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. Answers are
generated from retrieved passages and may be wrong &mdash; follow the citations.
Built by <a href="https://amiheines.com">Ami Heines</a>.
</footer>
<script>
const out=document.getElementById('out'),qi=document.getElementById('q');
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function go(mode){
  const q=qi.value.trim(); if(!q) return;
  out.innerHTML='<p class=sub>'+(mode==='ask'?'Retrieving and generating&hellip; (up to a minute)':'Searching&hellip;')+'</p>';
  try{
    const r=await fetch('/api/'+(mode==='ask'?'ask':'search')+'?q='+encodeURIComponent(q));
    const d=await r.json();
    if(d.error){out.innerHTML='<p class=err>'+esc(d.error)+'</p>';return}
    let h='';
    if(d.answer) h+='<div class=answer>'+esc(d.answer)+'</div>';
    const list=d.results||d.sources||[];
    h+=list.map(x=>'<div class=hit><a href="'+x.url+'" target=_blank rel=noopener>'+esc(x.title)+'</a>'
      +(x.similarity!==undefined?' <span class=score>'+x.similarity+'</span>':'')
      +(x.extract?'<p>'+esc(x.extract)+'&hellip;</p>':'')+'</div>').join('');
    h+='<p class=sub>'+d.took_ms+' ms</p>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<p class=err>Request failed.</p>'}
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();go('ask')};
document.getElementById('bSearch').onclick=()=>go('search');
</script></body></html>"""


@app.get("/")
def index():
    return PAGE.replace("__MAXQ__", str(MAX_Q)), 200, {"Content-Type": "text/html; charset=utf-8"}
