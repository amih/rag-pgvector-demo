"""Load PubMed XML (baseline, then daily update files) into Postgres. CPU only.

Resumable: each file is upserted + recorded in `loaded_files` in one transaction,
so a kill/reboot just redoes the file it was on. Files are applied strictly in
order (baseline 0001..N, then updates) because later files revise earlier ones.
Parsing runs in a small worker pool; commits stay ordered via imap.

A revised record drops its old embedding so embed_pubmed.py redoes it.
"""
import gzip
import os
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

from common import RAW, SCHEMA, connect

WORKERS = int(os.environ.get("LOAD_WORKERS", 6))

# Priority 1 = what this index is mainly for; embedded first. Matched against
# MeSH descriptors + chemical names (lowercased substrings) and title/abstract.
MESH_HINTS = (
    "plants, medicinal", "plant extracts", "plant preparations", "phytochemicals",
    "phytotherapy", "herbal medicine", "medicine, traditional", "plants, edible",
    "spices", "seeds", "fruit", "vegetables", "dietary supplements", "functional food",
    "anthelmintic", "helminth", "nematod", "cestod", "trematod", "schistosom",
    "antiparasitic", "parasitic diseases", "tubulin", "microtubule",
    "antineoplastic agents, phytogenic", "alkaloids", "flavonoids", "polyphenols",
    "terpenes", "oils, volatile", "isothiocyanates", "sulfur compounds",
    "garlic", "nigella", "syzygium", "curcum", "berberine", "artemisinin",
    "benzimidazoles", "mebendazole", "albendazole", "fenbendazole", "ivermectin",
)
TEXT_RE = re.compile(
    r"anthelmint|helminth|nematicid|vermifug|deworm|tubulin|microtubul|phytochem|"
    r"plant extract|essential oil|benzophenanthridine|sanguinarine|chelerythrine|"
    r"chelidonine|thymoquinone|nigella|allicin|ajoene|garlic|eugenol|clove|"
    r"curcumin|berberine|sulforaphane|isothiocyanate|noscapine|podophyllotoxin|"
    r"combretastatin|colchicine|benzimidazole|mebendazole|fenbendazole|albendazole",
    re.I)


def _text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def _year(art):
    for path in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year"):
        y = art.findtext(path)
        if y and y[:4].isdigit():
            return int(y[:4])
    m = re.search(r"\d{4}", art.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")
    return int(m.group()) if m else None


def parse_file(path):
    """-> (name, rows, deleted_pmids)."""
    rows, deleted = [], []
    with gzip.open(path, "rb") as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "DeleteCitation":
                deleted += [int(p.text) for p in el.findall("PMID")]
                el.clear()
            elif el.tag == "PubmedArticle":
                mc = el.find("MedlineCitation")
                pmid_el = mc.find("PMID")
                if pmid_el.get("Version", "1") != "1":
                    el.clear(); continue
                art = mc.find("Article")
                title = _text(art.find("ArticleTitle"))
                parts = []
                for a in art.findall("Abstract/AbstractText"):
                    t = _text(a)
                    if t:
                        lab = a.get("Label")
                        parts.append(f"{lab}: {t}" if lab and lab.upper() != "UNLABELLED" else t)
                abstract = "\n".join(parts) or None
                ids = {i.get("IdType"): (i.text or "").strip()
                       for i in el.findall("PubmedData/ArticleIdList/ArticleId")}
                doi = ids.get("doi") or next(
                    (e.text for e in art.findall("ELocationID") if e.get("EIdType") == "doi"), None)
                mesh = [d.text for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName") if d.text]
                chems = [c.text for c in mc.findall("ChemicalList/Chemical/NameOfSubstance") if c.text]
                ptypes = [p.text for p in art.findall("PublicationTypeList/PublicationType") if p.text]
                tags = " | ".join(mesh + chems).lower()
                prio = 1 if (any(h in tags for h in MESH_HINTS)
                             or TEXT_RE.search(title) or TEXT_RE.search(abstract or "")) else 2
                rows.append((int(pmid_el.text), title or "(no title)", abstract,
                             art.findtext("Journal/Title"), _year(art), doi,
                             ids.get("pmc"), mesh, chems, ptypes, prio))
                el.clear()
    return os.path.basename(path), rows, deleted


def pending_files(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM loaded_files")
        done = {r[0] for r in cur.fetchall()}
    out = []
    for sub in ("baseline", "updatefiles"):
        d = os.path.join(RAW, "pubmed", sub)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith(".xml.gz") and f not in done]
    return out


COLS = "pmid,title,abstract,journal,year,doi,pmcid,mesh,chemicals,pubtypes,priority"


def apply(conn, name, rows, deleted):
    # Duplicate PMIDs inside one file: last one wins.
    rows = list({r[0]: r for r in rows}.values())
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS stage (LIKE pubmed_articles) ON COMMIT DELETE ROWS")
        with cur.copy(f"COPY stage ({COLS}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
        # Revised records lose their embedding (content may have changed).
        cur.execute("DELETE FROM pubmed_emb e USING stage s, pubmed_articles a "
                    "WHERE e.pmid = s.pmid AND a.pmid = s.pmid "
                    "AND (a.title, coalesce(a.abstract,'')) IS DISTINCT FROM (s.title, coalesce(s.abstract,''))")
        sets = ",".join(f"{c}=EXCLUDED.{c}" for c in COLS.split(",")[1:])
        cur.execute(f"INSERT INTO pubmed_articles ({COLS}) SELECT {COLS} FROM stage "
                    f"ON CONFLICT (pmid) DO UPDATE SET {sets}")
        if deleted:
            cur.execute("DELETE FROM pubmed_emb WHERE pmid = ANY(%s)", (deleted,))
            cur.execute("DELETE FROM pubmed_articles WHERE pmid = ANY(%s)", (deleted,))
        cur.execute("INSERT INTO loaded_files (name, records, deleted) VALUES (%s,%s,%s)",
                    (name, len(rows), len(deleted)))
    conn.commit()


def main():
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    files = pending_files(conn)
    print(f"[load] {len(files)} files pending", flush=True)
    with Pool(WORKERS) as pool:
        for i, (name, rows, deleted) in enumerate(pool.imap(parse_file, files), 1):
            apply(conn, name, rows, deleted)
            p1 = sum(r[-1] == 1 for r in rows)
            print(f"[{i}/{len(files)}] {name}: {len(rows):,} records ({p1:,} priority), "
                  f"{len(deleted)} deleted", flush=True)
    conn.close()


if __name__ == "__main__":
    sys.exit(main())
