"""Embedding backends.

  st     -> sentence-transformers on the GPU. This is the FAST bulk path.
            (The GTX 1080 is Pascal / SM 6.1. HuggingFace TEI's GPU images
            require Turing / SM 7.5+, so they will NOT run on this card —
            drive torch directly instead. torch's CUDA wheels do support 6.1.)
  ollama -> the already-running Ollama. Zero setup, but ~32 chunks/s (per this
            box's benchmark) because of per-request overhead, not raw compute.

Both prepend nomic's task prefixes (search_document: / search_query:), which the
model was trained to expect. Skipping them measurably hurts retrieval.
"""
import json
import urllib.request
from config import cfg


class OllamaBackend:
    def __init__(self):
        self.endpoint = cfg.ollama_url.rstrip("/") + "/api/embed"
        self.model = cfg.embed_model

    def _post(self, inputs):
        body = json.dumps({"model": self.model, "input": inputs}).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["embeddings"]

    def embed_docs(self, texts):
        return self._post([cfg.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self._post([cfg.query_prefix + text])[0]


class STBackend:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self._torch = torch
        self.m = SentenceTransformer(cfg.embed_model, trust_remote_code=True,
                                     device=cfg.device)
        self.m.max_seq_length = 512  # ~2k chars; covers our chunk size w/ headroom

    def _encode(self, texts):
        return self.m.encode(texts, batch_size=cfg.embed_batch,
                             normalize_embeddings=cfg.normalize,
                             convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_docs(self, texts):
        try:
            return self._encode([cfg.doc_prefix + t for t in texts])
        except self._torch.cuda.OutOfMemoryError:
            # Transient VRAM pressure (e.g. Ollama grabbed the GPU mid-run).
            # Free, split the batch, and retry — so a 6-day run survives contention.
            self._torch.cuda.empty_cache()
            if len(texts) == 1:
                raise
            mid = len(texts) // 2
            return self.embed_docs(texts[:mid]) + self.embed_docs(texts[mid:])

    def embed_query(self, text):
        return self.m.encode([cfg.query_prefix + text],
                             normalize_embeddings=cfg.normalize,
                             convert_to_numpy=True)[0].tolist()


def get_backend():
    return STBackend() if cfg.embed_backend == "st" else OllamaBackend()
