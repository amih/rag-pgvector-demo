"""Deep research: several passes over the five databases for a question that links
two subjects, e.g. "tubulin effect of nigella sativa".

One search cannot answer that kind of question: the papers about the plant rarely
name the protein, and the papers about the protein rarely name the plant. The
link is usually indirect: a compound of the plant that was tested on the protein,
or a mechanism both sides share. So, in the spirit of Swanson's "ABC" literature-
based discovery, a job runs these steps:

  1. plan      the language model splits the question into its subjects
               (A = nigella sativa, C = tubulin), with scientific names and kinds
  2. expand    for each subject, no language model:
               - organism: its compounds from LOTUS and Dr. Duke
               - target or process: compounds tested on it in ChEMBL assays, and
                 molecules whose mechanism names it
               - passages about it from Wikipedia and PubMed
  3. bridge    - compound overlap: an exact join of A's compounds with C's
               - direct pairs: every A compound searched together with C in
                 ChEMBL, Dr. Duke and LOTUS (both words must match)
               - the model reads both sides and names shared mechanisms and the
                 compounds most worth checking; each becomes a targeted search
  4. follow up the model looks at what was found and asks for a few more searches
  5. report    the model writes a report from the evidence only: direct evidence,
               indirect links, gaps, hypotheses. Every claim cites [n].

The language model is either the local one (through gpu-queue, at the priority
the user picked) or Claude through the Anthropic API.

Jobs run in background threads; their progress and result are kept as JSON files
in ASK_DEEP_DIR so a restart does not lose finished reports.
"""
import json
import os
import re
import threading
import time
import traceback
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import psycopg

from config import cfg
from sources import LITDB_URL, SOURCES, search_sources

DEEP_DIR = os.environ.get("ASK_DEEP_DIR", "deep")
LOCAL_MODEL = os.environ.get("DEEP_LOCAL_MODEL", cfg.chat_model)
LOCAL_CTX = int(os.environ.get("DEEP_NUM_CTX", 12288))
CLAUDE_MODEL = os.environ.get("DEEP_CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("DEEP_CLAUDE_EFFORT", "medium")
LLM_TIMEOUT = float(os.environ.get("DEEP_LLM_TIMEOUT", 1800))   # includes the GPU queue wait
WORKERS = int(os.environ.get("DEEP_WORKERS", 2))
# Claude Opus 5.5 list prices, USD per million tokens (input, output).
CLAUDE_PRICE = (float(os.environ.get("DEEP_CLAUDE_IN_PRICE", 4)),
                float(os.environ.get("DEEP_CLAUDE_OUT_PRICE", 20)))


def _deep_ollama_url():
    """Same gpu-queue proxy as Answer, under its own client name, so the status
    page shows deep-research jobs apart from quick answers."""
    url = os.environ.get("DEEP_OLLAMA_URL")
    if url:
        return url.rstrip("/")
    base = cfg.ollama_url.rstrip("/")
    return re.sub(r"/ollama/[^/]+$", "/ollama/ask-deep", base)


OLLAMA_URL = _deep_ollama_url()
PRIORITIES = ("interactive", "background")


def engines():
    out = {"local": f"Local model ({LOCAL_MODEL}) on the home GPU"}
    if os.environ.get("ANTHROPIC_API_KEY"):
        out["claude"] = f"Claude ({CLAUDE_MODEL}) via the Anthropic API"
    return out


# ---------------------------------------------------------------- language model
_claude_client = None


def _local(prompt, schema, priority, max_out):
    body = {"model": LOCAL_MODEL, "prompt": prompt, "stream": False,
            "options": {"num_ctx": LOCAL_CTX, "temperature": 0.2, "num_predict": max_out}}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "X-GPU-Priority": priority})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        return json.loads(r.read())["response"], (0, 0)


def _claude(prompt, schema, max_out):
    import anthropic
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(timeout=600, max_retries=3)
    output_config = {"effort": CLAUDE_EFFORT}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    msg = _claude_client.beta.messages.create(
        model=CLAUDE_MODEL, max_tokens=max_out, output_config=output_config,
        # On a policy decline the API re-runs the request on a fallback model.
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt}])
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, (msg.usage.input_tokens, msg.usage.output_tokens)


class Job:
    def __init__(self, d):
        self.d = d
        self._lock = threading.Lock()

    def log(self, text):
        with self._lock:
            self.d["steps"].append({"t": round(time.time() - self.d["started"], 1), "text": text})
        save(self)

    def llm(self, prompt, schema=None, max_out=2000):
        if self.d["engine"] == "claude":
            text, (tin, tout) = _claude(prompt, schema, max_out)
            with self._lock:
                u = self.d["usage"]
                u["input_tokens"] += tin
                u["output_tokens"] += tout
                u["usd"] = round((u["input_tokens"] * CLAUDE_PRICE[0]
                                  + u["output_tokens"] * CLAUDE_PRICE[1]) / 1e6, 3)
        else:
            text, _ = _local(prompt, schema, self.d["priority"], max_out)
        with self._lock:
            self.d["usage"]["llm_calls"] += 1
        if schema is None:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}


# ---------------------------------------------------------------- job store
_jobs = {}
_store_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="deep")


def _path(job_id):
    return os.path.join(DEEP_DIR, job_id + ".json")


def save(job):
    os.makedirs(DEEP_DIR, exist_ok=True)
    with job._lock:
        data = json.dumps(job.d)
    tmp = _path(job.d["id"]) + ".tmp"
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, _path(job.d["id"]))


def get(job_id):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id or ""):
        return None
    with _store_lock:
        if job_id in _jobs:
            return _jobs[job_id].d
    try:
        with open(_path(job_id)) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if d.get("status") in ("queued", "running"):   # the process died under it
        d["status"], d["error"] = "error", "Interrupted by a restart. Run it again."
    return d


def list_for(owner, limit=20):
    out = []
    if os.path.isdir(DEEP_DIR):
        files = sorted((f for f in os.listdir(DEEP_DIR) if f.endswith(".json")),
                       key=lambda f: os.path.getmtime(os.path.join(DEEP_DIR, f)), reverse=True)
        for f in files:
            d = get(f[:-5])
            if d and d["owner"] == owner:
                out.append({k: d.get(k) for k in ("id", "q", "engine", "priority", "status",
                                                  "created", "finished")})
                if len(out) >= limit:
                    break
    return out


def count_today(owner=None, engine=None):
    """Jobs started today (by this owner / with this engine), from memory and disk."""
    today = time.strftime("%Y-%m-%d")
    n = 0
    if os.path.isdir(DEEP_DIR):
        for f in os.listdir(DEEP_DIR):
            if not f.endswith(".json"):
                continue
            p = os.path.join(DEEP_DIR, f)
            if time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(p))) != today:
                continue
            d = get(f[:-5])
            if (d and d.get("created", "").startswith(today)
                    and (owner is None or d["owner"] == owner)
                    and (engine is None or d["engine"] == engine)):
                n += 1
    return n


def active_for(owner):
    with _store_lock:
        return any(j.d["owner"] == owner and j.d["status"] in ("queued", "running")
                   for j in _jobs.values())


def start(owner, q, sources, year, engine, priority):
    job = Job({
        "id": uuid.uuid4().hex[:12], "owner": owner, "q": q, "sources": sources, "year": year,
        "engine": engine, "priority": priority, "status": "queued",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "started": time.time(), "finished": None,
        "steps": [], "plan": None, "bridges": [], "evidence": [], "report": None, "error": None,
        "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
        "model": engines().get(engine, engine),
    })
    with _store_lock:
        _jobs[job.d["id"]] = job
    save(job)
    _pool.submit(_run, job)
    return job.d["id"]


def _run(job):
    job.d["status"] = "running"
    job.d["started"] = time.time()
    try:
        research(job)
        job.d["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        job.d["status"], job.d["error"] = "error", f"{type(e).__name__}: {e}"[:300]
    job.d["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    job.log("Finished." if job.d["status"] == "done" else "Stopped: " + job.d["error"])
    with _store_lock:
        _jobs.pop(job.d["id"], None)


# ---------------------------------------------------------------- database helpers
def _db():
    conn = psycopg.connect(LITDB_URL, autocommit=True)
    conn.execute("SET statement_timeout = '8s'")
    return conn


# A name we can search on: not an IUPAC string or a formula.
_GOOD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-αβγδ]{2,40}$")


def _usable(name):
    if re.fullmatch(r"[A-Z]{14}-[A-Z]{8,10}-[A-Z]", name):   # an InChIKey, not a name
        return False
    return bool(_GOOD_NAME.match(name)) and sum(c.isdigit() for c in name) <= 2 \
        and name.count("-") <= 2


def _split_list(body, label):
    """'... | Chemicals: A, B, C | ...' -> [A, B, C]."""
    m = re.search(re.escape(label) + r"\s*(.*?)(?:\s\|\s|$)", body)
    if not m:
        return []
    return [x.strip(" …") for x in m.group(1).split(", ") if x.strip(" …")]


def organism_compounds(names):
    """Compounds reported from an organism in LOTUS and Dr. Duke."""
    found, docs = [], []
    with _db() as conn, conn.cursor() as cur:
        for n in names:
            cur.execute("""SELECT source, title, body, url FROM src_docs
                           WHERE (source = 'lotus' AND kind = 'organism' AND lower(title) = lower(%s))
                              OR (source = 'duke' AND kind = 'plant' AND title ILIKE %s)
                           LIMIT 4""", (n, n + "%"))
            for src, title, body, url in cur.fetchall():
                label = "natural products:" if src == "lotus" else "Chemicals:"
                found += _split_list(body, label)
                docs.append((src, title, body, url))
            if docs:
                break
    seen, out = set(), []
    for x in found:
        k = x.lower()
        if k not in seen and _usable(x):
            seen.add(k)
            out.append(x)
    return out, docs


# Nutrients and minerals that every organism has: matching them to a target says
# nothing (ALANINE turns up in "alanine residue 227 of beta3-tubulin").
_COMMON = set("""water ash fat fiber protein carbohydrates kcal calories starch sugars sucrose
glucose fructose maltose lactose cellulose alanine arginine asparagine aspartic-acid cysteine
cystine glutamic-acid glutamine glycine histidine isoleucine leucine lysine methionine
phenylalanine proline serine threonine tryptophan tyrosine valine calcium potassium sodium
magnesium iron zinc copper manganese phosphorus selenium sulfur chromium cobalt nickel boron
aluminum silicon iodine palmitic-acid stearic-acid oleic-acid linoleic-acid linolenic-acid
alpha-linolenic-acid myristic-acid lauric-acid arachidic-acid behenic-acid niacin riboflavin
thiamin ascorbic-acid beta-carotene resin cholesterol saponins tannin tannins mucilage
essential-oil oil gum wax alkaloids flavonoids sterols phytosterols""".split())
COMMON_IN = int(os.environ.get("DEEP_COMMON_IN", 1000))   # found in more organisms = skip


def rank_specific(compounds):
    """Drops nutrients and compounds found almost everywhere, and orders the rest
    rarest first: a compound found in 11 organisms says more about one plant than
    one found in 800."""
    names = [c for c in compounds if c.lower().replace(" ", "-") not in _COMMON]
    if not names:
        return []
    n_org = {}
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT lower(title), (substring(body from 'Found in (\\d+) organisms'))::int
                       FROM src_docs WHERE source = 'lotus' AND kind = 'compound'
                         AND lower(title) = ANY(%s)""", ([c.lower() for c in names],))
        for t, n in cur.fetchall():
            n_org[t] = max(n_org.get(t, 0), n or 0)
    names = [c for c in names if n_org.get(c.lower(), 0) <= COMMON_IN]
    return sorted(names, key=lambda c: n_org.get(c.lower(), 50))


def target_compounds(names, limit=3000):
    """Compounds tested on a target (ChEMBL assays) or whose mechanism names it,
    with how many assays each appears in."""
    cnt = Counter()
    mech = []
    with _db() as conn, conn.cursor() as cur:
        tsq = " | ".join(f"({_phrase(n)})" for n in names)
        cur.execute(f"""SELECT body FROM src_docs
                        WHERE source = 'chembl' AND kind = 'assay'
                          AND tsv @@ to_tsquery('english', %s) LIMIT {limit}""", (tsq,))
        for (body,) in cur.fetchall():
            for c in _split_list(body, "Compounds:"):
                cnt[c.lower()] += 1
        cur.execute("""SELECT title, body, url FROM src_docs
                       WHERE source = 'chembl' AND kind = 'molecule'
                         AND body ILIKE ANY(%s) LIMIT 200""",
                    ([f"%Mechanism:%{n}%" for n in names],))
        for title, body, url in cur.fetchall():
            cnt[title.lower()] += 5
            mech.append((title, body, url))
    return cnt, mech


def _phrase(text):
    """A Postgres tsquery string that matches the words of `text` in a row."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    return " <-> ".join(words) if words else "''"


def pair_hits(compounds, other_names, per=2, cap=80):
    """Records in ChEMBL, Dr. Duke and LOTUS that name a compound AND the other subject."""
    other = " | ".join(f"({_phrase(n)})" for n in other_names)
    out = []
    with _db() as conn, conn.cursor() as cur:
        for c in compounds[:cap]:
            try:
                cur.execute("""SELECT source, kind, ext_id, title, body, url FROM src_docs
                               WHERE tsv @@ (phraseto_tsquery('english', %s)
                                             && to_tsquery('english', %s))
                               LIMIT %s""", (c, other, per))
            except psycopg.errors.QueryCanceled:
                continue
            for src, kind, ext_id, title, body, url in cur.fetchall():
                out.append({"compound": c, "source": src, "kind": kind, "ext_id": ext_id,
                            "title": title, "body": body, "url": url})
    return out


def duke_activities(compounds, cap=40):
    """Dr. Duke's listed biological activities per compound (mechanism hints)."""
    out = {}
    with _db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT title, body FROM src_docs
                       WHERE source = 'duke' AND kind = 'chemical' AND lower(title) = ANY(%s)""",
                    ([c.lower() for c in compounds[:cap]],))
        for title, body in cur.fetchall():
            acts = _split_list(body, "Activities:")
            if acts:
                out[title] = acts[:25]
    return out


# ---------------------------------------------------------------- evidence
class Evidence:
    def __init__(self):
        self.items, self._keys = [], {}

    def add(self, src, title, text, url, meta="", why=""):
        key = (url or "") + "|" + title
        if key in self._keys:
            return self._keys[key]
        n = len(self.items) + 1
        self.items.append({"n": n, "source": src, "label": SOURCES.get(src, {}).get("label", src),
                           "title": title, "url": url, "meta": meta, "why": why,
                           "text": (text or "")[:1500]})
        self._keys[key] = n
        return n

    def add_hits(self, found, why):
        ns = []
        for s, hits in found.items():
            if isinstance(hits, list):
                for h in hits:
                    ns.append(self.add(s, h["title"], h["text"], h["url"], h.get("meta", ""), why))
        return ns


def _brief(ev, ns, chars=350):
    return "\n".join(f"[{ev.items[n-1]['n']}] ({ev.items[n-1]['label']}) "
                     f"{ev.items[n-1]['title']}: {ev.items[n-1]['text'][:chars]}" for n in ns)


# ---------------------------------------------------------------- schemas
def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}
PLAN_SCHEMA = _obj({
    "subjects": {"type": "array", "items": _obj({
        "name": _STR, "scientific_name": _STR,
        "kind": {"type": "string", "enum": ["organism", "compound", "protein_or_target",
                                            "process_or_disease", "other"]},
        "synonyms": _STRS})},
    "focus": _STR})
BRIDGE_SCHEMA = _obj({
    "mechanisms": {"type": "array", "items": _obj({"name": _STR, "why": _STR, "query": _STR})},
    "compounds_to_check": _STRS})
FOLLOW_SCHEMA = _obj({"queries": _STRS, "note": _STR})


# ---------------------------------------------------------------- the research
def research(job):
    d = job.d
    q, sources, year = d["q"], d["sources"], d["year"]
    text_sources = [s for s in sources if s in ("wiki", "pubmed")] or ["pubmed"]
    ev = Evidence()

    # 1. plan
    job.log("Step 1 of 5: splitting the question into its subjects.")
    plan = job.llm(
        "You plan a literature search over Wikipedia, PubMed abstracts, ChEMBL (compounds and "
        "assays), Dr. Duke's phytochemical database and LOTUS (natural products by organism).\n"
        "Split the research question into its subjects: usually two, at most three (e.g. a plant "
        "and a protein). The effect, target, process or disease asked about is always a subject "
        "of its own: 'anthelmintic effect of allicin from garlic' has three subjects (allicin, "
        "garlic, anthelmintic activity), while 'tubulin effect of nigella sativa' has two "
        "(nigella sativa, tubulin). Never add a subject that only restates another. For each give the plain name, the scientific name (Latin binomial for "
        "organisms, standard name for proteins/compounds; repeat the name if none), its kind, and "
        "up to 4 synonyms or common alternative names. 'focus' is one sentence saying what "
        "relation between the subjects the user wants to know.\n\n"
        f"Question: {q}", PLAN_SCHEMA, 600)
    subjects = [s for s in (plan.get("subjects") or []) if s.get("name")][:3]
    if not subjects:
        subjects = [{"name": q, "scientific_name": q, "kind": "other", "synonyms": []}]
    d["plan"] = {"subjects": subjects, "focus": plan.get("focus") or q}
    job.log("Subjects: " + "; ".join(f"{s['name']} ({s['kind'].replace('_', ' ')})"
                                     for s in subjects))

    # 2. expand each subject
    job.log("Step 2 of 5: what each database knows about each subject.")
    side = []
    for s in subjects:
        names = list(dict.fromkeys(x for x in [s["scientific_name"], s["name"], *s["synonyms"]]
                                   if x and x.strip()))
        info = {"s": s, "names": names, "compounds": [], "tested": Counter(), "ev": []}
        found = search_sources(s["name"] if s["name"] == s["scientific_name"]
                               else f"{s['name']} ({s['scientific_name']})",
                               sources, 5, year)
        info["ev"] += ev.add_hits(found, f"about {s['name']}")
        if s["kind"] == "organism":
            comps, docs = organism_compounds(names)
            comps = rank_specific(comps)
            info["compounds"] = comps
            for src, title, body, url in docs:
                info["ev"].append(ev.add(src, title, body, url, "compound list",
                                         f"compounds of {s['name']}"))
            job.log(f"{s['name']}: {len(comps)} named compounds in LOTUS / Dr. Duke "
                    f"(nutrients and near-universal ones left out), e.g. {', '.join(comps[:8])}.")
        elif s["kind"] == "compound":
            info["compounds"] = names
        if s["kind"] in ("protein_or_target", "process_or_disease", "other"):
            tested, mech = target_compounds(names)
            info["tested"] = tested
            for title, body, url in mech[:5]:
                info["ev"].append(ev.add("chembl", title, body, url, "mechanism",
                                         f"acts on {s['name']}"))
            job.log(f"{s['name']}: {len(tested)} compounds tested on it or acting on it "
                    f"in ChEMBL.")
        side.append(info)

    # 3. bridges
    job.log("Step 3 of 5: looking for links between the subjects.")
    bridges = []
    for a in side:
        for c in side:
            if a is c or not a["compounds"]:
                continue
            # 3a. exact overlap: A's compounds that ChEMBL lists as tested on C
            overlap = [x for x in a["compounds"] if c["tested"].get(x.lower())]
            for x in overlap:
                bridges.append({"type": "compound tested on target", "a": a["s"]["name"],
                                "c": c["s"]["name"], "via": x,
                                "detail": f"{x} is a compound of {a['s']['name']} and appears in "
                                          f"{c['tested'][x.lower()]} ChEMBL records about "
                                          f"{c['s']['name']}"})
            # 3b. records naming a compound of A together with C
            hits = pair_hits(overlap + [x for x in a["compounds"] if x not in overlap], c["names"])
            for h in hits:
                n = ev.add(h["source"], h["title"], h["body"], h["url"],
                           f"{h['kind']} · {h['ext_id']}",
                           f"names {h['compound']} and {c['s']['name']}")
                bridges.append({"type": "record names both", "a": a["s"]["name"],
                                "c": c["s"]["name"], "via": h["compound"], "n": n,
                                "detail": f"{SOURCES[h['source']]['label']} {h['kind']} "
                                          f"'{h['title'][:120]}' names {h['compound']} and "
                                          f"{c['s']['name']}"})
            job.log(f"{a['s']['name']} → {c['s']['name']}: {len(overlap)} shared compounds, "
                    f"{len(hits)} database records naming a compound and {c['s']['name']}.")
    d["bridges"] = bridges
    terms = [[t.lower() for t in x["names"] + x["compounds"][:40] if len(t) > 3] for x in side]

    # 3c. the model names shared mechanisms and the compounds worth checking
    acts = {}
    for a in side:
        if a["compounds"]:
            acts.update(duke_activities(a["compounds"]))
    if len(side) >= 2:
        sides_txt = "\n\n".join(
            f"== {x['s']['name']} ==\n{_brief(ev, x['ev'][:10], 300)}"
            + (f"\nCompounds: {', '.join(x['compounds'][:80])}" if x["compounds"] else "")
            + (f"\nCompounds most tested on it: "
               f"{', '.join(k for k, _ in x['tested'].most_common(40))}" if x["tested"] else "")
            for x in side)
        acts_txt = "\n".join(f"- {k}: {', '.join(v)}" for k, v in list(acts.items())[:25])
        found_txt = "\n".join(f"- {b['detail']}" for b in bridges[:30]) or "(none)"
        br = job.llm(
            f"Research question: {q}\nFocus: {d['plan']['focus']}\n\n"
            "Below is what the databases say about each subject separately, the biological "
            "activities Dr. Duke lists for the compounds, and the direct links already found.\n"
            "1) Name up to 6 mechanisms, pathways, processes or intermediate molecules that could "
            "connect the subjects (e.g. mitotic arrest, ROS, apoptosis, a shared protein). Prefer "
            "ones supported by the text on BOTH sides. For each give a short reason and one "
            "search query (a few words naming both sides) to look for the link in PubMed.\n"
            "2) From the compound lists, pick up to 5 compounds most worth checking directly "
            "against the other subject (well-known, abundant, or bioactive ones).\n\n"
            f"{sides_txt}\n\nCompound activities (Dr. Duke):\n{acts_txt or '(none)'}\n\n"
            f"Direct links already found:\n{found_txt}", BRIDGE_SCHEMA, 1200)
        mechs = (br.get("mechanisms") or [])[:6]
        checks = [c for c in (br.get("compounds_to_check") or []) if c][:5]
        d["mechanisms"] = mechs
        job.log("Possible shared mechanisms: " + (", ".join(m["name"] for m in mechs) or "none"))
        others = [x["s"]["name"] for x in side if x["s"]["kind"] != "organism"] or \
                 [side[-1]["s"]["name"]]
        queries = [(m["query"], f"mechanism: {m['name']}") for m in mechs if m.get("query")]
        queries += [(f"{c} {others[0]}", f"compound check: {c}") for c in checks
                    if c.lower() != others[0].lower()]
        _run_queries(job, ev, queries, text_sources, year)

    # 4. follow-up round
    job.log("Step 4 of 5: checking for gaps and following up.")
    titles = "\n".join(f"[{e['n']}] ({e['label']}, {e['why']}) {e['title']}" for e in ev.items[:120])
    fu = job.llm(
        f"Research question: {q}\nFocus: {d['plan'].get('focus', q)}\n\n"
        f"Evidence gathered so far (titles only):\n{titles}\n\n"
        f"Links found: {len(bridges)} database records; mechanisms considered: "
        f"{', '.join(m['name'] for m in d.get('mechanisms', [])) or 'none'}.\n\n"
        "What is still missing to answer the question? Give up to 4 more PubMed/Wikipedia "
        "search queries (a few words each) that would fill the most important gaps, and a "
        "one-sentence note on the gap.", FOLLOW_SCHEMA, 500)
    more = [x for x in (fu.get("queries") or []) if x][:4]
    if fu.get("note"):
        job.log("Gap: " + fu["note"])
    _run_queries(job, ev, [(x, "follow-up") for x in more], text_sources, year)

    # 5. report
    job.log("Step 5 of 5: writing the report.")
    d["evidence"] = ev.items
    pick = _select(ev, bridges, terms, 60 if d["engine"] == "claude" else 18)
    chars = 900 if d["engine"] == "claude" else 450
    ctx = "\n\n".join(f"[{e['n']}] ({e['label']}; found as: {e['why']}) {e['title']}"
                      + (f" — {e['meta']}" if e["meta"] else "") + f"\n{e['text'][:chars]}"
                      for e in pick)
    facts = "\n".join(f"- {b['detail']}" + (f" [{b['n']}]" if b.get("n") else "")
                      for b in bridges[:40]) or "- none found"
    d["report"] = job.llm(
        f"Write a research report on: {q}\nFocus: {d['plan'].get('focus', q)}\n\n"
        "Use ONLY the evidence below. Cite every claim with its number, like [3] or [3][7]. "
        "Never invent a citation or a fact. Say plainly when the evidence is thin.\n"
        "Structure (markdown, '## ' headings, short paragraphs and bullet lists):\n"
        "## Short answer — 2-4 sentences.\n"
        "## Direct evidence — studies or records that name both subjects together.\n"
        "## Indirect links — through a compound, a shared mechanism or an intermediate "
        "molecule. For each, spell out the chain (A → B → C) and how strong each step is.\n"
        "## Gaps and caveats — what was not found, dose/model limits (cell line, animal, human).\n"
        "## Hypotheses worth testing — clearly labelled as hypotheses, not findings.\n\n"
        f"Links found by database joins:\n{facts}\n\nEvidence:\n{ctx}",
        None, 6000 if d["engine"] == "claude" else 1800)
    d["evidence"] = ev.items
    save(job)


def _run_queries(job, ev, queries, sources, year):
    if not queries:
        return
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [(why, query, pool.submit(search_sources, query, sources, 4, year))
                for query, why in queries]
        for why, query, fut in futs:
            try:
                n = len(ev.add_hits(fut.result(timeout=60), f"{why} — “{query}”"))
            except Exception as e:
                job.log(f"Search “{query}” failed ({type(e).__name__}).")
                continue
            job.log(f"Searched “{query}”: {n} passages.")


def _select(ev, bridges, terms, k):
    """What the report gets to read, best first:
      0  passages that name both subjects (or a compound of one and the other)
      1  database records found by the bridge joins (a few)
      2  targeted searches (mechanisms, compound checks), 3 follow-ups, 4 background."""
    bridge_ns = list(dict.fromkeys(b["n"] for b in bridges if b.get("n")))[:max(3, k // 6)]

    def names_both(e):
        text = (e["title"] + " " + e["text"]).lower()
        return len(terms) >= 2 and sum(any(t in text for t in ts) for ts in terms) >= 2

    def order(e):
        why = e["why"]
        if names_both(e) and e["source"] in ("wiki", "pubmed"):
            tier = 0
        elif e["n"] in bridge_ns:
            tier = 1
        elif why.startswith(("mechanism", "compound check")):
            tier = 2
        elif why.startswith("follow-up"):
            tier = 3
        else:
            tier = 4
        return (tier, e["n"])
    return sorted(ev.items, key=order)[:k]
