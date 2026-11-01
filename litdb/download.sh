#!/usr/bin/env bash
# Mirror the open literature/natural-product datasets into $RAW (HDD).
# Idempotent: re-run any time; wget -N / -c only fetches what changed.
# Polite: one connection, rate-capped, idle I/O priority (set in the unit).
set -euo pipefail
RAW=${RAW:-/mnt/nvme3/litdb/raw}
RATE=${RATE:-10m}
W="wget -q -c --limit-rate=$RATE --tries=20 --waitretry=30"
# get FILE URL: fetch once (these files are versioned by name or rarely change).
get() { [ -s "$1" ] || $W -O "$1.part" "$2" && { [ -s "$1" ] || mv "$1.part" "$1"; }; }
mkdir -p "$RAW"/{pubmed/baseline,pubmed/updatefiles,chembl,duke,lotus}

echo "[pubmed] baseline + daily updates"
for d in baseline updatefiles; do
  # (wget -r is blocked by NCBI's robots.txt, so list the index ourselves.)
  base="https://ftp.ncbi.nlm.nih.gov/pubmed/$d/"
  ( cd "$RAW/pubmed/$d" && curl -s "$base" | grep -oE 'pubmed[0-9]+n[0-9]+\.xml\.gz(\.md5)?"' | tr -d '"' | sort -u |
      sed "s#^#$base#" | wget -q -N --limit-rate="$RATE" --tries=20 --waitretry=30 -i - )
  # Verify checksums; a bad file is deleted so the next run refetches it.
  ( cd "$RAW/pubmed/$d" && for m in *.md5; do
      f=${m%.md5}; [ -f "$f" ] || continue
      want=$(grep -oE '[0-9a-f]{32}' "$m"); have=$(md5sum "$f" | cut -c1-32)
      [ "$want" = "$have" ] || { echo "  bad md5: $f (removed)"; rm -f "$f"; }
    done )
done

echo "[chembl] latest SQLite"
CH=$(curl -s https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ | grep -oE 'chembl_[0-9]+_sqlite\.tar\.gz' | head -1)
( cd "$RAW/chembl" && get "$CH" "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/$CH" \
    && { ls -d chembl_*/ >/dev/null 2>&1 || tar -I pigz -xf "$CH"; } )

echo "[duke] Dr. Duke's Phytochemical and Ethnobotanical DB (USDA, CC0)"
( cd "$RAW/duke" && get Duke-Source-CSV.zip https://ndownloader.figshare.com/files/43363335 \
    && get DataDictionary.csv https://ndownloader.figshare.com/files/43363338 \
    && unzip -oq Duke-Source-CSV.zip )

echo "[lotus] latest frozen dataset (Zenodo)"
LOT=$(curl -sL https://zenodo.org/api/records/5794106/versions/latest)
echo "$LOT" | python3 -c '
import sys, json
for f in json.load(sys.stdin)["files"]:
    if f["key"].endswith(".csv.gz"): print(f["key"], f["links"]["self"])' |
while read -r name url; do ( cd "$RAW/lotus" && get "$name" "$url" ); done

echo "[done] $(du -sh "$RAW" | cut -f1) in $RAW"
