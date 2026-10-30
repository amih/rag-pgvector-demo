# litdb — local PubMed + natural-product databases

| What | Where | Size (approx.) |
|---|---|---|
| PubMed baseline + updates (XML) | `$RAW/pubmed` (HDD) | ~45 GB gz |
| ChEMBL (SQLite) — bioactivity, incl. tubulin assays | `$RAW/chembl` | ~30 GB |
| Dr. Duke's Phytochemical & Ethnobotanical DB (CSV) | `$RAW/duke` | small |
| LOTUS (structure–organism pairs, CSV) | `$RAW/lotus` | ~0.5 GB |
| Postgres `litdb`: `pubmed_articles`, `pubmed_emb` (nomic-768, halfvec) | wiki-rag container :5463 (NVMe) | ~130 GB with index |

Pipeline: `litdb-fetch.service` (download.sh → load_pubmed.py, weekly timer) and
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
