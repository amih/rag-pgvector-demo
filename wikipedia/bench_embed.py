"""Measure real embedding throughput for the configured backend, then project
the full-Wikipedia time. Run this FIRST — it turns the estimate into a number.

  EMBED_BACKEND=st     python bench_embed.py     # fast GPU path
  EMBED_BACKEND=ollama python bench_embed.py     # baseline (~32 ch/s here)
"""
import time
from config import cfg
from zim_source import ZimSource
from chunker import chunk
from embed_backends import get_backend

SAMPLE = 512


def main():
    src = ZimSource()
    chunks = []
    for _, _, text in src.iter_from(src.entry_count // 3):  # sample mid-corpus
        chunks.extend(chunk(text, cfg.chunk_size, cfg.chunk_overlap))
        if len(chunks) >= SAMPLE:
            break
    chunks = chunks[:SAMPLE]

    backend = get_backend()
    backend.embed_docs(chunks[:8])  # warmup (loads model onto GPU)

    t = time.time()
    for i in range(0, len(chunks), cfg.embed_batch):
        backend.embed_docs(chunks[i:i + cfg.embed_batch])
    dt = time.time() - t
    rate = len(chunks) / dt

    print(f"backend={cfg.embed_backend}  model={cfg.embed_model}  batch={cfg.embed_batch}")
    print(f"{len(chunks)} chunks in {dt:.2f}s  =>  {rate:.1f} chunks/s")
    print("projected full-Wikipedia embedding time:")
    for label, total in [("prose-clean ~30M", 30_000_000), ("raw-strip ~45M", 45_000_000)]:
        hrs = total / rate / 3600
        print(f"  {label:>18}: {hrs:6.1f} h  ({hrs/24:.1f} days)")


if __name__ == "__main__":
    main()
