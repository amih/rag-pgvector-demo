"""Load ChEMBL, Dr. Duke and LOTUS into one keyword-searchable table, `src_docs`.

PubMed gets vector search (embed_pubmed.py). These three are structured
databases: names of compounds, plants, targets and assays. Exact words matter
more than meaning there, so each record becomes one short text document with a
Postgres full-text index (title weighted above body). One row per:

  chembl/molecule  named molecule + synonyms, mechanisms, indications
  chembl/assay     assay description, target, organism, paper, named compounds tested
  duke/chemical    chemical, its biological activities, which plants contain it
  duke/plant       plant, common names, folk uses (with country), chemicals
  lotus/compound   natural product, chemical class, organisms it was found in
  lotus/organism   organism and the natural products reported from it

Each source is replaced in one transaction, so the web app never sees half a
source. src_lexemes holds per-source word counts; the search uses it to drop
words that match too much of a source (see search_sources in the web app).

  python load_structured.py [chembl] [duke] [lotus]      # default: all three
"""
import csv
import glob
import gzip
import sqlite3
import sys
import time
from collections import defaultdict

from common import RAW, connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS src_docs (
    id     bigserial PRIMARY KEY,
    source text NOT NULL,          -- chembl | duke | lotus
    kind   text NOT NULL,          -- molecule | assay | chemical | plant | compound | organism
    ext_id text NOT NULL,
    title  text NOT NULL,
    body   text NOT NULL,
    url    text,
    tsv    tsvector GENERATED ALWAYS AS (
             setweight(to_tsvector('english', title), 'A') ||
             setweight(to_tsvector('english', body), 'B')) STORED
);
CREATE INDEX IF NOT EXISTS src_docs_tsv ON src_docs USING gin (tsv);
CREATE INDEX IF NOT EXISTS src_docs_source ON src_docs (source);
CREATE TABLE IF NOT EXISTS src_lexemes (
    source text NOT NULL,
    lexeme text NOT NULL,
    ndoc   int  NOT NULL,
    PRIMARY KEY (source, lexeme)
);
"""

MAX_BODY = 6000


def clip(parts, limit=MAX_BODY):
    s = " | ".join(p for p in parts if p)
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + " …"


def uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------- ChEMBL
def chembl_docs():
    path = glob.glob(f"{RAW}/chembl/chembl_*/chembl_*_sqlite/chembl_*.db")
    if not path:
        raise SystemExit("ChEMBL SQLite file not found under " + RAW)
    db = sqlite3.connect(f"file:{sorted(path)[-1]}?mode=ro", uri=True)
    q = db.execute

    syn = defaultdict(list)
    for molregno, s in q("SELECT molregno, synonyms FROM molecule_synonyms"):
        syn[molregno].append(s)
    mech = defaultdict(list)
    for molregno, moa, action, target in q(
            "SELECT m.molregno, m.mechanism_of_action, m.action_type, t.pref_name "
            "FROM drug_mechanism m LEFT JOIN target_dictionary t USING (tid)"):
        mech[molregno].append(f"{moa or ''} ({(action or '').lower()}; target {target or '?'})")
    ind = defaultdict(list)
    for molregno, heading, phase in q(
            "SELECT molregno, mesh_heading, max_phase_for_ind FROM drug_indication"):
        ind[molregno].append(f"{heading} (phase {phase:g})" if phase is not None else heading)

    for molregno, name, chembl_id, phase, mtype, natural, vet in q(
            "SELECT molregno, pref_name, chembl_id, max_phase, molecule_type, "
            "natural_product, veterinary FROM molecule_dictionary "
            "WHERE pref_name IS NOT NULL OR molregno IN (SELECT molregno FROM molecule_synonyms)"):
        title = name or (syn[molregno][0] if syn[molregno] else chembl_id)
        flags = [mtype or "",
                 f"max phase {phase:g}" if phase else "",
                 "natural product" if natural == 1 else "",
                 "veterinary" if vet == 1 else ""]
        body = clip([", ".join(f for f in flags if f),
                     "Synonyms: " + ", ".join(uniq(syn[molregno])[:40]) if syn[molregno] else "",
                     "Mechanism: " + "; ".join(uniq(mech[molregno])) if mech[molregno] else "",
                     "Indications: " + ", ".join(uniq(ind[molregno])[:40]) if ind[molregno] else ""])
        yield ("molecule", chembl_id, title, body,
               f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}")

    print("  chembl: collecting named compounds per assay …", flush=True)
    tested = defaultdict(list)
    for assay_id, name in q(
            "SELECT DISTINCT a.assay_id, m.pref_name FROM activities a "
            "JOIN molecule_dictionary m ON m.molregno = a.molregno "
            "WHERE m.pref_name IS NOT NULL"):
        if len(tested[assay_id]) < 12:
            tested[assay_id].append(name)

    for (assay_id, chembl_id, desc, organism, strain, cell, target, t_org,
         d_title, journal, year) in q(
            "SELECT a.assay_id, a.chembl_id, a.description, a.assay_organism, "
            "a.assay_strain, a.assay_cell_type, t.pref_name, t.organism, "
            "d.title, d.journal, d.year "
            "FROM assays a LEFT JOIN target_dictionary t USING (tid) "
            "LEFT JOIN docs d USING (doc_id)"):
        if not desc:
            continue
        body = clip([
            desc,
            f"Target: {target}" + (f" ({t_org})" if t_org else "") if target else "",
            "Organism: " + " ".join(x for x in (organism, strain) if x) if organism else "",
            f"Cell: {cell}" if cell else "",
            "Compounds: " + ", ".join(tested[assay_id]) if tested[assay_id] else "",
            "Paper: " + " ".join(str(x) for x in (d_title, journal, year) if x) if d_title else "",
        ])
        yield ("assay", chembl_id, desc[:240], body,
               f"https://www.ebi.ac.uk/chembl/explore/assay/{chembl_id}")


# ---------------------------------------------------------------- Dr. Duke
def read_csv(name):
    with open(f"{RAW}/duke/{name}", newline="", encoding="latin-1") as f:
        yield from csv.DictReader(f)


def duke_docs():
    parts = {r["PPCO"]: r["PPNA"] for r in read_csv("PARTS.csv")}
    taxa = {r["FNFNUM"]: r for r in read_csv("FNFTAX.csv")}
    common = defaultdict(list)
    for r in read_csv("COMMON_NAMES.csv"):
        common[r["FNFNUM"]].append(r["CNNAM"])

    acts = defaultdict(list)
    for r in read_csv("AGGREGAC.csv"):
        a = r["ACTIVITY"]
        if r["DOSAGE"]:
            a += f" ({r['DOSAGE']})"
        acts[r["CHEM"]].append(a)

    in_plant = defaultdict(list)          # chem -> "Allium sativum (Garlic) bulb 1500-27800 ppm"
    plant_chems = defaultdict(list)       # taxon -> chems
    for r in read_csv("FARMACY_NEW.csv"):
        t = taxa.get(r["FNFNUM"])
        if not t:
            continue
        taxon = t["TAXON"]
        cn = common[r["FNFNUM"]]
        lo, hi = r["AMT_OR_LO"] or r["AMT_LO"], r["AMT_HI"] or r["AMT_OR_HI"]
        amt = f"{lo}-{hi} ppm" if lo and hi else (f"{lo or hi} ppm" if lo or hi else "")
        part = parts.get(r["PPCO"], r["PPCO"])
        in_plant[r["CHEM"]].append(
            " ".join(x for x in (taxon, f"({cn[0]})" if cn else "", part.lower(), amt) if x))
        plant_chems[taxon].append(r["CHEM"])

    for chem in uniq(list(acts) + list(in_plant)):
        body = clip(["Activities: " + ", ".join(uniq(acts[chem])) if acts[chem] else "",
                     "Found in: " + "; ".join(uniq(in_plant[chem])) if in_plant[chem] else ""])
        yield ("chemical", chem, chem.title(), body, "https://phytochem.nal.usda.gov/")

    uses = defaultdict(list)
    ecommon = defaultdict(list)
    family = {}
    for r in read_csv("ETHNOBOT.csv"):
        taxon = r["TAXON"]
        uses[taxon].append(r["ACTIVITY"] + (f" ({r['COUNTRY']})" if r["COUNTRY"] else ""))
        if r["CNAME"]:
            ecommon[taxon].append(r["CNAME"])
        family.setdefault(taxon, r["FAMILY"])
    by_taxon = {t["TAXON"]: num for num, t in taxa.items()}
    for taxon in uniq(list(plant_chems) + list(uses)):
        num = by_taxon.get(taxon)
        names = uniq(common[num] + ecommon[taxon]) if num else uniq(ecommon[taxon])
        fam = (taxa[num]["FAMILY"] if num else "") or family.get(taxon, "")
        title = taxon + (f" ({', '.join(names[:3])})" if names else "")
        body = clip([f"Family {fam}" if fam else "",
                     "Common names: " + ", ".join(names) if names else "",
                     "Folk uses: " + ", ".join(uniq(uses[taxon])) if uses[taxon] else "",
                     "Chemicals: " + ", ".join(uniq(plant_chems[taxon]))
                     if plant_chems[taxon] else ""])
        yield ("plant", taxon, title, body, "https://phytochem.nal.usda.gov/")


# ---------------------------------------------------------------- LOTUS
def lotus_docs():
    path = sorted(glob.glob(f"{RAW}/lotus/*_frozen_metadata.csv.gz"))[-1]
    chembl_names = {}
    cpath = glob.glob(f"{RAW}/chembl/chembl_*/chembl_*_sqlite/chembl_*.db")
    if cpath:   # LOTUS names are often IUPAC-ish; ChEMBL knows the common name.
        db = sqlite3.connect(f"file:{sorted(cpath)[-1]}?mode=ro", uri=True)
        for key, name in db.execute(
                "SELECT s.standard_inchi_key, m.pref_name FROM compound_structures s "
                "JOIN molecule_dictionary m USING (molregno) WHERE m.pref_name IS NOT NULL"):
            chembl_names[key] = name

    comp, orgs, refs = {}, defaultdict(list), defaultdict(set)
    org_family, org_wd, org_comps = {}, {}, defaultdict(list)
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            key = r["structure_inchikey"]
            if key not in comp:
                comp[key] = r
            org = r["organism_name"]
            orgs[key].append(org)
            if r["reference_doi"]:
                refs[key].add(r["reference_doi"])
            org_family.setdefault(org, r["organism_taxonomy_06family"])
            org_wd.setdefault(org, r["organism_wikidata"])
            org_comps[org].append(key)

    def cname(key):
        r = comp[key]
        return (chembl_names.get(key) or r["structure_nameTraditional"]
                or r["structure_nameIupac"] or key)

    for key, r in comp.items():
        name = cname(key)
        klass = " / ".join(x for x in (r["structure_taxonomy_npclassifier_01pathway"],
                                       r["structure_taxonomy_npclassifier_02superclass"],
                                       r["structure_taxonomy_npclassifier_03class"]) if x)
        o = uniq(orgs[key])
        body = clip([
            f"{r['structure_molecular_formula']}" + (f", {klass}" if klass else ""),
            f"Other name: {r['structure_nameTraditional']}"
            if r["structure_nameTraditional"] and r["structure_nameTraditional"] != name else "",
            f"Found in {len(o)} organisms: " + ", ".join(
                f"{x} ({org_family[x]})" if org_family.get(x) else x for x in o[:60]),
            f"{len(refs[key])} references",
        ])
        yield ("compound", key, name, body, r["structure_wikidata"] or None)

    for org, keys in org_comps.items():
        k = uniq(keys)
        names = uniq(cname(x) for x in k)
        body = clip([f"Family {org_family[org]}" if org_family.get(org) else "",
                     f"{len(k)} natural products: " + ", ".join(names[:150])])
        yield ("organism", org, org, body, org_wd.get(org) or None)


# ---------------------------------------------------------------- load
LOADERS = {"chembl": chembl_docs, "duke": duke_docs, "lotus": lotus_docs}


def load(conn, source):
    t0 = time.time()
    n = 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM src_docs WHERE source = %s", (source,))
        cur.execute("DELETE FROM src_lexemes WHERE source = %s", (source,))
        with cur.copy("COPY src_docs (source, kind, ext_id, title, body, url) FROM STDIN") as cp:
            for kind, ext_id, title, body, url in LOADERS[source]():
                cp.write_row((source, kind, ext_id, title[:500], body, url))
                n += 1
                if n % 200_000 == 0:
                    print(f"  {source}: {n:,} …", flush=True)
        print(f"  {source}: counting words …", flush=True)
        cur.execute("""INSERT INTO src_lexemes (source, lexeme, ndoc)
                       SELECT %s, word, ndoc FROM ts_stat(
                         'SELECT tsv FROM src_docs WHERE source = ''' || %s || '''')""",
                    (source, source))
    conn.commit()
    print(f"{source}: {n:,} documents in {time.time() - t0:.0f}s", flush=True)


def main():
    sources = sys.argv[1:] or list(LOADERS)
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()
        for s in sources:
            load(conn, s)
        conn.execute("ANALYZE src_docs")


if __name__ == "__main__":
    main()
