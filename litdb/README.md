# litdb — local PubMed + natural-product databases

| What | Where | Size (approx.) |
|---|---|---|
| PubMed baseline + updates (XML) | `$RAW/pubmed` (NVMe) | ~45 GB gz |
| ChEMBL (SQLite) — bioactivity, incl. tubulin assays | `$RAW/chembl` | ~30 GB |
| Dr. Duke's Phytochemical & Ethnobotanical DB (CSV) | `$RAW/duke` | small |
| LOTUS (structure–organism pairs, CSV) | `$RAW/lotus` | ~0.5 GB |
| Postgres `litdb`: `pubmed_articles`, `pubmed_emb` (nomic-768, halfvec) | wiki-rag container :5463 (NVMe) | ~130 GB with index |
| Postgres `litdb`: `src_docs` (ChEMBL, Dr. Duke, LOTUS as text + full-text index) | same | 2.3M rows |

`$RAW` defaults to `/mnt/nvme3/litdb/raw`. Keep it off any disk that is the target of
an `rsync --delete` backup, and exclude it from backups: everything here can be
downloaded again.

Pipeline: `litdb-fetch.service` (download.sh → load_pubmed.py → load_structured.py, weekly timer) and
`litdb-embed.service` (embed_pubmed.py, runs continuously, **yields the GPU** —
see gpu_gate.py). Natural-products/parasite/tubulin papers (priority 1) are
embedded first.

    sudo cp litdb-*.service litdb-*.timer /etc/systemd/system/ && sudo systemctl daemon-reload
    sudo systemctl enable --now litdb-fetch.timer litdb-embed.service
    sudo systemctl start --no-block litdb-fetch.service      # first run now
    journalctl -u litdb-fetch -f ; journalctl -u litdb-embed -f
    touch /mnt/nvme3/litdb/PAUSE   # pause embedding; rm to resume
    python gpu_gate.py             # why is it paused?

When embedding is done: `python build_index.py` (hours, CPU), then
`python search.py "query" [--prio1] [--from 2010] [--mesh Anthelmintics]`.

## ChEMBL, Dr. Duke and LOTUS — `load_structured.py`

These are lists of names (compounds, plants, targets, assays), so they get keyword
search rather than embeddings. Each record becomes one short document in
`src_docs` — a ChEMBL molecule with its synonyms, mechanisms and indications; a
ChEMBL assay with its target, organism, paper and the named compounds tested; a
Dr. Duke chemical with its activities and the plants that contain it; a Dr. Duke
plant with its common names, folk uses and chemicals; a LOTUS compound with the
organisms it was found in; a LOTUS organism with its compounds. `src_lexemes`
keeps per-source word counts, used to weight rare words above common ones.

    python load_structured.py            # all three, ~5 min; or: duke | lotus | chembl

The web front end (`../wikipedia/ask_web.py`, `sources.py`) searches these next to
Wikipedia and PubMed.
