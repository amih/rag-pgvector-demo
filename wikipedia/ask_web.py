"""Read-only web front end: one question searched across Wikipedia, PubMed,
ChEMBL, Dr. Duke and LOTUS (see sources.py). Signed-in users only (auth.py).

Two endpoints, deliberately asymmetric, because they cost wildly different things:

  /api/search   vector + keyword search, no LLM. Under a few seconds. Cheap.
  /api/ask      retrieve + generate an answer with [n] citations. Seconds, and it
                occupies the whole GPU while it runs.

That asymmetry is the entire protection design. A request-per-second limit is the
wrong control for /api/ask: the binding resource is not request rate, it is the
single GPU, which serves exactly one generation at a time. Ten clients each
politely sending one request per second still queue forever. So:

  * nginx  enforces the crude flood limits (req/s, concurrent conns, body size).
  * here   enforces the expensive-resource limits — one generation at a time, a
           bounded wait for a slot, a per-user hourly cap, and a global daily
           budget. Denial-of-wallet controls, with the GPU as the wallet.

Everything here is in-process memory: this is a single-worker service by design
(one GPU, one generation), so there is nothing to share. Counters reset on
restart, which is fine — they exist to bound abuse, not to bill anyone.

Run:  waitress-serve --listen=127.0.0.1:3600 ask_web:app
"""
import json
import os
import threading
import time
import urllib.request
from collections import defaultdict, deque

from flask import Flask, jsonify, request, session

import auth
import deep
from config import cfg
from sources import KEYWORD_SOURCES, SOURCES, search_sources, warm_up

app = Flask(__name__)
auth.init_app(app)
warm_up()

MAX_Q = int(os.environ.get("ASK_MAX_Q", 300))              # chars
GEN_SLOT_WAIT = float(os.environ.get("ASK_GEN_WAIT", 25))  # s to wait for the GPU
GEN_TIMEOUT = float(os.environ.get("ASK_GEN_TIMEOUT", 120))
USER_ASK_PER_HOUR = int(os.environ.get("ASK_IP_HOURLY", 20))
USER_SEARCH_PER_HOUR = int(os.environ.get("ASK_IP_SEARCH_HOURLY", 120))
DAILY_ASK_BUDGET = int(os.environ.get("ASK_DAILY_BUDGET", 300))
DEEP_USER_DAILY = int(os.environ.get("DEEP_USER_DAILY", 10))      # jobs per user per day
DEEP_CLAUDE_DAILY = int(os.environ.get("DEEP_CLAUDE_DAILY", 20))  # Claude jobs, all users

# How many passages from each database go into the answer's context. Every
# selected database gets a share, so the answer can draw on all of them.
CONTEXT_PER_SOURCE = {"wiki": 2, "pubmed": 4, "chembl": 2, "duke": 2, "lotus": 1}
PASSAGE_CHARS = 1200

# One generation at a time. The 1080 has 8GB; llama3.1:8b fills it. A second
# concurrent generation does not run twice as slow, it thrashes.
_gpu = threading.BoundedSemaphore(1)
_lock = threading.Lock()
_hits = defaultdict(deque)          # (user, kind) -> timestamps, trimmed to 1h
_day = {"stamp": None, "asks": 0}


def _take_slot(who: str, kind: str, per_hour: int):
    """Sliding-window per-user counter. Returns None if allowed, else an error."""
    now = time.time()
    with _lock:
        q = _hits[(who, kind)]
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
            return "Daily answer budget reached. Search still works."
        _day["asks"] += 1
    return None


def _params():
    q = (request.args.get("q") or "").strip()
    if not q:
        return None, "Missing ?q="
    if len(q) > MAX_Q:
        return None, f"Question too long (max {MAX_Q} characters)."
    src = [s for s in (request.args.get("sources") or "").split(",") if s in SOURCES]
    try:
        year = int(request.args.get("from") or 0) or None
    except ValueError:
        year = None
    return {"q": q, "sources": src or list(SOURCES), "year": year}, None


def _results(found):
    """{source: hits} -> list of sections in fixed order, with labels."""
    out = []
    for s, meta in SOURCES.items():
        if s not in found:
            continue
        hits = found[s]
        sec = {"source": s, "label": meta["label"], "about": meta["about"],
               "match": "keyword" if s in KEYWORD_SOURCES else "meaning"}
        if isinstance(hits, dict):
            sec.update(error=f"{meta['label']} search failed ({hits['error']}).", hits=[])
        else:
            sec["hits"] = [{**h, "extract": h["text"][:600]} for h in hits]
            for h in sec["hits"]:
                h.pop("text", None)
        out.append(sec)
    return out


def generate(question, passages):
    context = "\n\n".join(
        f"[{i+1}] ({p['label']}) {p['title']}: {p['text'][:PASSAGE_CHARS]}"
        for i, p in enumerate(passages))
    prompt = ("Answer the question using ONLY the context below. The passages come from "
              "different databases, named in parentheses. Cite sources as [n]. "
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
    p, err = _params()
    if err:
        return jsonify(error=err), 400
    if (err := _take_slot(auth.user_key(), "search", USER_SEARCH_PER_HOUR)):
        return jsonify(error=err), 429
    k = min(int(request.args.get("k", 5) or 5), 15)
    t0 = time.time()
    found = search_sources(p["q"], p["sources"], k, p["year"])
    return jsonify(query=p["q"], took_ms=round((time.time() - t0) * 1000, 1),
                   sections=_results(found))


@app.get("/api/ask")
def api_ask():
    p, err = _params()
    if err:
        return jsonify(error=err), 400
    if (err := _take_slot(auth.user_key(), "ask", USER_ASK_PER_HOUR)):
        return jsonify(error=err), 429
    if (err := _take_daily_budget()):
        return jsonify(error=err), 429

    t0 = time.time()
    found = search_sources(p["q"], p["sources"], 5, p["year"])
    passages = []
    for s in SOURCES:
        hits = found.get(s)
        if isinstance(hits, list):
            passages += [{**h, "source": s, "label": SOURCES[s]["label"]}
                         for h in hits[:CONTEXT_PER_SOURCE[s]]]
    if not passages:
        return jsonify(error="No results in the selected databases."), 404

    # Bounded wait: fail fast rather than let a queue build behind the GPU.
    if not _gpu.acquire(timeout=GEN_SLOT_WAIT):
        return jsonify(error="Busy — one answer is generated at a time. "
                             "Try again shortly, or use Search only."), 503
    try:
        answer = generate(p["q"], passages)
    except Exception as e:
        return jsonify(error=f"Generation failed: {type(e).__name__}"), 502
    finally:
        _gpu.release()

    return jsonify(
        query=p["q"], answer=answer, took_ms=round((time.time() - t0) * 1000, 1),
        sources=[{"n": i + 1, "source": x["source"], "label": x["label"],
                  "title": x["title"], "url": x["url"], "meta": x.get("meta", "")}
                 for i, x in enumerate(passages)],
        sections=_results(found))


@app.post("/api/deep")
def api_deep_start():
    if not request.is_json:   # JSON only: a cross-site form cannot send it
        return jsonify(error="Send JSON."), 415
    b = request.get_json(silent=True) or {}
    q = (b.get("q") or "").strip()
    if not q:
        return jsonify(error="Missing question."), 400
    if len(q) > MAX_Q:
        return jsonify(error=f"Question too long (max {MAX_Q} characters)."), 400
    engine = b.get("engine") if b.get("engine") in deep.engines() else "local"
    priority = b.get("priority") if b.get("priority") in deep.PRIORITIES else "background"
    src = [s for s in (b.get("sources") or []) if s in SOURCES] or list(SOURCES)
    try:
        year = int(b.get("from") or 0) or None
    except (TypeError, ValueError):
        year = None
    who = auth.user_key()
    if deep.active_for(who):
        return jsonify(error="You already have a deep research running. "
                             "Wait for it to finish."), 429
    if deep.count_today(owner=who) >= DEEP_USER_DAILY:
        return jsonify(error=f"Limit: {DEEP_USER_DAILY} deep research runs per day."), 429
    if engine == "claude" and deep.count_today(engine="claude") >= DEEP_CLAUDE_DAILY:
        return jsonify(error="Today's Claude budget is used up. Use the local model."), 429
    return jsonify(id=deep.start(who, q, src, year, engine, priority))


@app.get("/api/deep")
def api_deep_list():
    u = auth.current_user()
    all_users = request.args.get("all") == "1" and bool(u.get("admin"))
    return jsonify(jobs=deep.list_for(auth.user_key(), all_users=all_users))


@app.delete("/api/deep/<job_id>")
def api_deep_delete(job_id):
    d = deep.get(job_id)
    u = auth.current_user()
    if not d or (d["owner"] != auth.user_key() and not u.get("admin")):
        return jsonify(error="Not found."), 404
    if not deep.delete(job_id):
        return jsonify(error="It is still running."), 409
    return jsonify(ok=True)


@app.get("/api/deep/<job_id>")
def api_deep_get(job_id):
    d = deep.get(job_id)
    u = auth.current_user()
    if not d or (d["owner"] != auth.user_key() and not u.get("admin")):
        return jsonify(error="Not found."), 404
    return jsonify({k: v for k, v in d.items() if k != "owner"})


@app.get("/api/me")
def api_me():
    u = auth.current_user()
    return jsonify(email=u["email"], name=session.get("name", ""), admin=bool(u.get("admin")),
                   sources=[{"id": k, **v} for k, v in SOURCES.items()],
                   engines=[{"id": k, "label": v} for k, v in deep.engines().items()])


@app.get("/healthz")
def healthz():
    with _lock:
        return jsonify(ok=True, asks_today=_day["asks"], budget=DAILY_ASK_BUDGET)


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Ask — Wikipedia, PubMed, ChEMBL, Dr. Duke, LOTUS</title>
<style>
:root{color-scheme:light dark;--fg:#1a1a1a;--bg:#fdfdfb;--muted:#5a5a5a;--line:#d8d5cc;--accent:#8a6a2f;--card:#f2f0ea;
--wiki:#3b6ea5;--pubmed:#2f7d5b;--chembl:#a0522d;--duke:#6b8e23;--lotus:#8b4f9e}
@media(prefers-color-scheme:dark){:root{--fg:#e8e6e1;--bg:#16161a;--muted:#a3a09a;--line:#33323a;--accent:#d4a94a;--card:#20202a;
--wiki:#7aa7d9;--pubmed:#6cc39b;--chembl:#e0956b;--duke:#a9c96a;--lotus:#c79ad6}}
*{box-sizing:border-box}
body{margin:0 auto;padding:1.25rem 1.25rem 5rem;max-width:48rem;background:var(--bg);color:var(--fg);
font:1.0625rem/1.65 Georgia,'Times New Roman',serif}
.top{display:flex;justify-content:flex-end;gap:1rem;font-size:.85rem;color:var(--muted)}
.top a{color:var(--accent)}
h1{font-size:1.6rem;margin:.5rem 0 .25rem}
p.sub{color:var(--muted);font-size:.95rem;margin:0 0 1.25rem}
form{display:flex;gap:.5rem;flex-wrap:wrap}
input[type=text]{flex:1 1 16rem;padding:.7rem .8rem;font:inherit;font-size:1rem;
background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}
input[type=number]{width:6rem;padding:.3rem .4rem;font:inherit;font-size:.9rem;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}
button{padding:.7rem 1.1rem;font:inherit;font-size:.95rem;cursor:pointer;border-radius:4px;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{border-color:var(--accent);color:var(--accent)}
fieldset{border:1px solid var(--line);border-radius:4px;margin:.9rem 0 0;padding:.5rem .8rem;font-size:.9rem}
legend{color:var(--muted);padding:0 .3rem}
fieldset label{display:inline-flex;align-items:center;gap:.3rem;margin:.15rem 1rem .15rem 0;cursor:pointer}
.opts{margin-top:.4rem;color:var(--muted)}
.hint{color:var(--muted);font-size:.85rem;margin:.6rem 0 0}
#out{margin-top:2rem}
.answer{background:var(--card);padding:1rem 1.2rem;border-radius:4px;border-left:3px solid var(--accent);white-space:pre-wrap}
.cites{font-size:.9rem;margin:.6rem 0 1.5rem;padding-left:0;list-style:none}
.cites li{margin:.2rem 0}.cites a{color:var(--accent)}
section{margin-top:1.8rem}
section h2{font-size:1.05rem;margin:0;display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap}
section h2 small{font-weight:normal;color:var(--muted);font-size:.8rem}
.badge{display:inline-block;font:600 .72rem/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.02em;
padding:.25rem .45rem;border-radius:3px;color:#fff;vertical-align:.1em}
.b-wiki{background:var(--wiki)}.b-pubmed{background:var(--pubmed)}.b-chembl{background:var(--chembl)}
.b-duke{background:var(--duke)}.b-lotus{background:var(--lotus)}
@media(prefers-color-scheme:dark){.badge{color:#111}}
.hit{border-top:1px solid var(--line);padding:.8rem 0}
.hit a{color:var(--accent);text-decoration:none;font-weight:600}
.hit p{margin:.3rem 0 0;font-size:.95rem}
.meta,.score{color:var(--muted);font-size:.8rem;font-family:ui-monospace,monospace}
.none{color:var(--muted);font-size:.9rem;border-top:1px solid var(--line);padding:.6rem 0}
.err{color:#b4451f}
details.how{margin:.9rem 0 0;font-size:.92rem;border:1px solid var(--line);border-radius:4px;padding:.5rem .8rem}
details.how summary{cursor:pointer;color:var(--accent)}
details.how ol{padding-left:1.3rem}details.how li{margin:.3rem 0}
.steps{font:.8rem/1.5 ui-monospace,monospace;color:var(--muted);background:var(--card);padding:.6rem .8rem;border-radius:4px;max-height:22rem;overflow:auto}
.steps b{color:var(--fg);font-weight:normal}
.report{background:var(--card);padding:.4rem 1.2rem 1rem;border-radius:4px;border-left:3px solid var(--accent)}
.report h2{font-size:1.1rem;margin:1.1rem 0 .3rem}.report ul{padding-left:1.2rem}
.report a.c{color:var(--accent);text-decoration:none;font-size:.85em}
.ev li{margin:.35rem 0;font-size:.9rem}.ev .why{color:var(--muted);font-size:.8rem}
table.hist{width:100%;border-collapse:collapse;font-size:.88rem}
table.hist td,table.hist th{text-align:left;padding:.45rem .4rem;border-top:1px solid var(--line);vertical-align:top}
table.hist th{color:var(--muted);font-weight:normal}table.hist a{color:var(--accent)}
table.hist button{padding:.15rem .5rem;font-size:.8rem}
.jobs{font-size:.88rem;margin-top:.6rem}.jobs a{color:var(--accent)}
footer{margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem;color:var(--muted);font-size:.85rem}
footer a{color:var(--accent)}
</style></head><body>
<div class=top><span id=who></span><a href="#history">Deep research history</a><a id=adm href=/admin hidden>Users</a><a href=/logout>Sign out</a></div>
<h1>Ask</h1>
<p class=sub>One question, five databases: Wikipedia, PubMed abstracts, ChEMBL, Dr. Duke's
phytochemical database and LOTUS natural products. Each database keeps its own top results.</p>
<form id=f>
  <input type=text id=q maxlength=__MAXQ__ placeholder="e.g. garlic allicin anthelmintic activity" autofocus>
  <button type=submit class=primary>Answer</button>
  <button type=button id=bSearch>Search only</button>
  <button type=button id=bDeep>Deep research</button>
</form>
<fieldset><legend>Search in</legend><div id=srcs></div>
<div class=opts><label>PubMed from year <input type=number id=yr min=1800 max=2100 placeholder=any></label></div>
</fieldset>
<fieldset><legend>Deep research</legend>
<div id=modes></div>
<div class=jobs id=jobs></div>
</fieldset>
<details class=how><summary>How deep research works</summary>
<p>For questions that link two subjects, like <i>tubulin effect of nigella sativa</i>. One search rarely
answers them: papers about the plant seldom name the protein, and papers about the protein seldom name the
plant. The link is usually indirect, through a compound of the plant or a mechanism both share. So deep
research makes several passes:</p>
<ol>
<li><b>Split the question.</b> The model finds the subjects: A = nigella sativa (an organism), C = tubulin (a protein).</li>
<li><b>Expand each subject, no model needed.</b> LOTUS and Dr. Duke list the plant's compounds (thymoquinone,
thymohydroquinone, nigellone, &alpha;-hederin&hellip;). ChEMBL lists compounds tested on tubulin in lab assays, and drugs
whose mechanism is tubulin. Wikipedia and PubMed add what is known about each.</li>
<li><b>Look for bridges.</b> An exact join: which of the plant's compounds were tested on tubulin. Then every compound
is searched together with tubulin in ChEMBL, Dr. Duke and LOTUS. The model reads both sides and names shared
mechanisms (mitotic arrest, ROS, apoptosis&hellip;) and the compounds most worth checking; each becomes a targeted
PubMed and Wikipedia search, e.g. <i>thymoquinone microtubule</i>.</li>
<li><b>Follow up.</b> The model looks at what was found and runs a few more searches for the gaps.</li>
<li><b>Report.</b> Written only from the evidence found, with numbered citations: short answer, direct evidence,
indirect links (A &rarr; B &rarr; C, with how strong each step is), gaps, and hypotheses &mdash; marked as hypotheses, not facts.</li>
</ol>
<p><b>Model.</b> The local model is free but small: it plans and links less well, and every call waits its turn on
the one GPU, so a run takes several minutes, longer at background priority. Claude reads far more evidence and reasons
better; it costs money per run (typically well under a dollar), shown with the report. Steps 2 and 3's database joins
are the same either way. <i>Now</i> puts the local model ahead of other work on the GPU; <i>in the background</i> lets it wait for an idle GPU.</p>
<h3 style="font-size:1rem;margin:1rem 0 .3rem">Where the data comes from</h3>
<p>All five databases are open or free to use, downloaded in full and kept on one home server, so a search never
leaves the machine (only Claude runs send the question and the found passages to Anthropic). They live in
PostgreSQL with the <a href="https://github.com/pgvector/pgvector">pgvector</a> extension, on NVMe disks.</p>
<ul>
<li><b>Wikipedia</b> (CC BY-SA 4.0): the full English Wikipedia as an offline Kiwix ZIM file (text only, June 2026).
Every article is cut into passages of about 1,200 characters with a small overlap: 35.8 million passages. Each is turned
into a 768-number vector by the open <i>nomic-embed-text v1.5</i> model on the server's GPU, stored at half precision
(<code>halfvec</code>) and indexed with an HNSW graph, so the closest passages in meaning come back in milliseconds.</li>
<li><b>PubMed</b> (free from the US National Library of Medicine): the yearly baseline files plus the daily update files,
fetched again every week. 29.6 million abstracts, each embedded with the same model (about 10 days of GPU time) and
indexed with HNSW (56&nbsp;GB). Your question gets the same kind of vector, so one search serves both.</li>
<li><b>ChEMBL</b> (EMBL-EBI, CC BY-SA 3.0): release 37, downloaded as a SQLite database. 2 million records: named
molecules with their synonyms, mechanisms and indications, and lab assays with their target, organism, paper and the
compounds tested.</li>
<li><b>Dr. Duke's Phytochemical and Ethnobotanical Databases</b> (USDA, US public domain): the CSV export. 43,000 records:
plants with their chemicals and folk uses by country, and chemicals with their biological activities and the plants that
contain them, with amounts.</li>
<li><b>LOTUS</b> (CC0, public domain): the frozen release from Zenodo. 265,000 records: natural products, their chemical
class and the organisms they were found in, and each organism with its compounds.</li>
</ul>
<p>Wikipedia and PubMed are prose, so they are matched by <i>meaning</i> (vectors). ChEMBL, Dr. Duke and LOTUS are mostly
names of compounds, plants and targets, where the exact word matters, so each record becomes one short text document
in a Postgres full-text index, and rare words (a compound name) count for more than common ones ("activity"). The
structured databases are reloaded every week, each in one transaction, so a search never sees half a database. The
links in step 3 use those same records: an organism's compound list, an assay's tested compounds.</p>
</details>
<p class=hint><b>Search only</b> lists matches from each database, no language model.
<b>Deep research</b> makes several passes to find direct and indirect links (see below).
<b>Answer</b> also has a local model write an answer from the top matches of every database you picked; one at a time, so it may queue.
Wikipedia and PubMed match by meaning; ChEMBL, Dr. Duke and LOTUS match the words you type, so use names of plants, compounds or effects.</p>
<div id=out></div>
<footer>
Sources: <a href="https://en.wikipedia.org">Wikipedia</a> (CC BY-SA 4.0),
<a href="https://pubmed.ncbi.nlm.nih.gov/">PubMed</a> (NLM),
<a href="https://www.ebi.ac.uk/chembl/">ChEMBL</a> (CC BY-SA 3.0),
<a href="https://phytochem.nal.usda.gov/">Dr. Duke's Phytochemical and Ethnobotanical Databases</a> (USDA),
<a href="https://lotus.naturalproducts.net/">LOTUS</a> (CC0).
Answers are generated from retrieved passages and may be wrong &mdash; follow the citations.
</footer>
<script>
const out=document.getElementById('out'),qi=document.getElementById('q'),yr=document.getElementById('yr');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let SOURCES=[];
const saved=JSON.parse(localStorage.getItem('ask_sources')||'null');
function picked(){return [...document.querySelectorAll('#srcs input:checked')].map(x=>x.value)}
fetch('/api/me').then(r=>r.json()).then(d=>{
  document.getElementById('who').textContent=d.email||'';
  if(d.admin) document.getElementById('adm').hidden=false;
  SOURCES=d.sources||[];
  document.getElementById('srcs').innerHTML=SOURCES.map(s=>'<label title="'+esc(s.about)+'"><input type=checkbox value='+s.id
    +((!saved||saved.includes(s.id))?' checked':'')+'> <span class="badge b-'+s.id+'">'+esc(s.label)+'</span></label>').join('');
  document.querySelectorAll('#srcs input').forEach(x=>x.onchange=()=>localStorage.setItem('ask_sources',JSON.stringify(picked())));
});
const saved_mode=localStorage.getItem('ask_deep_mode');
fetch('/api/me').then(r=>r.json()).then(d=>{
  const hasClaude=(d.engines||[]).some(e=>e.id==='claude');
  const M=[];
  if(hasClaude)M.push(['claude','<b>Claude</b> &mdash; best report, about $0.25 a run']);
  M.push(['local-interactive','<b>Local model, now</b> &mdash; free, goes ahead of other work on the home GPU']);
  M.push(['local-background','<b>Local model, in the background</b> &mdash; free, runs when the GPU is idle; check back later']);
  const pick=M.some(m=>m[0]===saved_mode)?saved_mode:M[0][0];
  document.getElementById('modes').innerHTML=M.map(m=>'<label><input type=radio name=mode value='+m[0]
    +(m[0]===pick?' checked':'')+'> '+m[1]+'</label>').join('<br>');
  document.querySelectorAll('input[name=mode]').forEach(x=>x.onchange=()=>localStorage.setItem('ask_deep_mode',x.value));
});
function loadJobs(){fetch('/api/deep').then(r=>r.json()).then(d=>{
  const j=d.jobs||[];document.getElementById('jobs').innerHTML=j.length?'Your recent runs: '+j.slice(0,8).map(x=>
    '<a href="#deep='+x.id+'">'+esc(x.q.slice(0,50))+'</a> <span class=opts>('+esc(x.status)+')</span>').join(' &middot; ')
    +' &middot; <a href="#history">all '+j.length+' &rarr;</a>':'';
})}
loadJobs();
function md(t){
  t=esc(t).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\w)/g,'$1<i>$2</i>').replace(/\[(\d+)((?:[-–,] ?\d+)*)\]/g,'<a class=c href="#ev$1">[$1$2]</a>');
  let h='',ul=false;
  for(const l of t.split('\n')){
    const m=l.match(/^\s*[-*] (.*)/);
    if(m){if(!ul){h+='<ul>';ul=true}h+='<li>'+m[1]+'</li>';continue}
    if(ul){h+='</ul>';ul=false}
    if(/^#{1,4} /.test(l))h+='<h2>'+l.replace(/^#+ /,'')+'</h2>';
    else if(l.trim())h+='<p>'+l+'</p>';
  }
  return h+(ul?'</ul>':'');
}
let poll=null;
function showDeep(id){
  clearTimeout(poll);
  fetch('/api/deep/'+id).then(r=>r.json()).then(d=>{
    if(d.error&&!d.id){out.innerHTML='<p class=err>'+esc(d.error)+'</p>';return}
    if(!qi.value)qi.value=d.q;
    let h='<h2>Deep research: '+esc(d.q)+'</h2><p class=sub>'+esc(d.model)+' &middot; '+esc(d.priority)+' priority &middot; '
      +esc(d.status)+(d.usage&&d.usage.llm_calls?' &middot; '+d.usage.llm_calls+' model calls':'')
      +(d.usage&&d.usage.usd?' &middot; $'+d.usage.usd.toFixed(3):'')+'</p>';
    if(d.report)h+='<div class=report>'+md(d.report)+'</div>';
    if(d.status==='error')h+='<p class=err>'+esc(d.error)+'</p>';
    if(d.status==='queued'||d.status==='running')h+='<p class=sub>Working&hellip; this page updates by itself; you can also leave and come back via "Your recent runs".</p>';
    h+='<details'+(d.report?'':' open')+'><summary>Steps</summary><div class=steps>'
      +d.steps.map(s=>'<div>'+s.t+'s <b>'+esc(s.text)+'</b></div>').join('')+'</div></details>';
    if(d.evidence&&d.evidence.length)h+='<details'+(d.report?' open':'')+'><summary>Evidence ('+d.evidence.length+')</summary><ol class=ev>'
      +d.evidence.map(e=>'<li id=ev'+e.n+' value='+e.n+'>'+badge(e.source,e.label)+' <a href="'+esc(e.url||'#')+'" target=_blank rel=noopener>'
        +esc(e.title)+'</a> <span class=why>'+esc(e.meta?e.meta+' · ':'')+esc(e.why)+'</span></li>').join('')+'</ol></details>';
    out.innerHTML=h;
    if(d.status==='queued'||d.status==='running')poll=setTimeout(()=>showDeep(id),3000);else loadJobs();
  }).catch(()=>{poll=setTimeout(()=>showDeep(id),6000)});
}
async function goDeep(){
  const q=qi.value.trim(); if(!q) return;
  const src=picked(); if(!src.length){out.innerHTML='<p class=err>Pick at least one database.</p>';return}
  const mode=(document.querySelector('input[name=mode]:checked')||{}).value||'local-background';
  const eng=mode==='claude'?'claude':'local',prio=mode==='local-interactive'?'interactive':'background';
  out.innerHTML='<p class=sub>Starting deep research&hellip;</p>';
  const r=await fetch('/api/deep',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({q,sources:src,from:yr.value||null,engine:eng,priority:prio})});
  if(r.status===401){location.reload();return}
  const d=await r.json();
  if(d.error){out.innerHTML='<p class=err>'+esc(d.error)+'</p>';return}
  location.hash='deep='+d.id;
}
let ADMIN=false;
fetch('/api/me').then(r=>r.json()).then(d=>{ADMIN=!!d.admin});
function showHistory(all){
  clearTimeout(poll);
  out.innerHTML='<p class=sub>Loading&hellip;</p>';
  fetch('/api/deep'+(all?'?all=1':'')).then(r=>r.json()).then(d=>{
    const j=d.jobs||[];
    let h='<h2>Deep research history</h2><p class=sub>Every run is saved until you delete it. Click a question to open its report.'
      +(ADMIN?' <a href="#history'+(all?'':'-all')+'">'+(all?'Show only mine':'Show all users')+'</a>':'')+'</p>';
    if(!j.length){out.innerHTML=h+'<p class=sub>No runs yet.</p>';return}
    h+='<table class=hist><tr><th>Date</th><th>Question</th><th>Model</th><th>Time</th><th>Cost</th><th></th></tr>'
      +j.map(x=>'<tr id=row'+x.id+'><td>'+esc((x.created||'').slice(0,16))+'</td><td><a href="#deep='+x.id+'">'+esc(x.q)+'</a>'
        +(x.status!=='done'?' <span class=opts>('+esc(x.status)+')</span>':'')
        +(all&&x.owner?'<div class=opts>'+esc(x.owner)+'</div>':'')+'</td><td>'+(x.engine==='claude'?'Claude':'Local')
        +(x.engine==='local'?'<div class=opts>'+esc(x.priority)+'</div>':'')+'</td><td>'+(x.seconds!=null?Math.round(x.seconds)+' s':'')
        +'</td><td>'+(x.usd?'$'+x.usd.toFixed(2):'')+'</td><td>'
        +(x.status==='queued'||x.status==='running'?'':'<button onclick="delRun(\''+x.id+'\')">Delete</button>')+'</td></tr>').join('')+'</table>';
    out.innerHTML=h;
  });
}
async function delRun(id){
  if(!confirm('Delete this deep research run? This cannot be undone.'))return;
  const r=await fetch('/api/deep/'+id,{method:'DELETE'});const d=await r.json();
  if(d.error){alert(d.error);return}
  const row=document.getElementById('row'+id);if(row)row.remove();loadJobs();
}
function route(){
  const m=location.hash.match(/deep=([0-9a-f]{12})/);if(m)return showDeep(m[1]);
  if(location.hash==='#history')return showHistory(false);
  if(location.hash==='#history-all')return showHistory(true);
}
const badge=(id,label)=>'<span class="badge b-'+id+'">'+esc(label)+'</span>';
function hit(h){
  return '<div class=hit>'+(h.url?'<a href="'+esc(h.url)+'" target=_blank rel=noopener>'+esc(h.title)+'</a>':'<b>'+esc(h.title)+'</b>')
    +' <span class=score>'+h.score+'</span>'
    +(h.meta?'<div class=meta>'+esc(h.meta)+'</div>':'')
    +(h.extract?'<p>'+esc(h.extract)+(h.extract.length>=600?'&hellip;':'')+'</p>':'')+'</div>';
}
function sections(list){
  return list.map(s=>'<section><h2>'+badge(s.source,s.label)+' <small>'+esc(s.about)+' &middot; matched by '+s.match+'</small></h2>'
    +(s.error?'<p class=err>'+esc(s.error)+'</p>':s.hits.length?s.hits.map(hit).join('')
      :'<div class=none>No match in '+esc(s.label)+(s.match==='keyword'?' &mdash; try plant, compound or effect names.':'.')+'</div>')
    +'</section>').join('');
}
async function go(mode){
  const q=qi.value.trim(); if(!q) return;
  const src=picked(); if(!src.length){out.innerHTML='<p class=err>Pick at least one database.</p>';return}
  out.innerHTML='<p class=sub>'+(mode==='ask'?'Searching and writing an answer&hellip; (up to a minute)':'Searching&hellip;')+'</p>';
  const u='/api/'+(mode==='ask'?'ask':'search')+'?q='+encodeURIComponent(q)+'&sources='+src.join(',')+(yr.value?'&from='+encodeURIComponent(yr.value):'');
  try{
    const r=await fetch(u); if(r.status===401){location.reload();return}
    const d=await r.json();
    if(d.error){out.innerHTML='<p class=err>'+esc(d.error)+'</p>';return}
    let h='';
    if(d.answer){
      h+='<div class=answer>'+esc(d.answer)+'</div><ul class=cites>'
        +d.sources.map(x=>'<li>['+x.n+'] '+badge(x.source,x.label)+' <a href="'+esc(x.url||'#')+'" target=_blank rel=noopener>'+esc(x.title)+'</a></li>').join('')+'</ul>';
    }
    h+=sections(d.sections||[]);
    h+='<p class=sub>'+d.took_ms+' ms</p>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<p class=err>Request failed.</p>'}
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();go('ask')};
document.getElementById('bSearch').onclick=()=>go('search');
document.getElementById('bDeep').onclick=goDeep;
window.addEventListener('hashchange',route);route();
</script></body></html>"""


@app.get("/")
def index():
    return PAGE.replace("__MAXQ__", str(MAX_Q)), 200, {"Content-Type": "text/html; charset=utf-8"}
