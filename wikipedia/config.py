"""Config from environment, with defaults tuned for the home server.

Load a .env with:  set -a; . ./.env; set +a   (or use `python -m dotenv`).
Every script imports `cfg` from here.
"""
import os


def _b(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


class cfg:
    # --- data source ---
    zim_path   = os.environ.get("ZIM_PATH",
                    "/mnt/hdd/wikipedia/kiwix/wikipedia_en_all_nopic_2026-06.zim")
    min_article_chars = int(os.environ.get("MIN_ARTICLE_CHARS", 200))

    # --- postgres ---
    # NOTE: point this at a Postgres whose data dir lives on NVMe (see README).
    db_url = os.environ.get("DATABASE_URL", "postgres://rag:rag@localhost:5462/rag")

    # --- vector storage ---
    # halfvec (fp16) halves storage + index size vs vector (fp32) at ~no recall loss.
    # Needs pgvector >= 0.7 (the pgvector/pgvector:pg16 image has it).
    vector_type = os.environ.get("VECTOR_TYPE", "halfvec")   # halfvec | vector
    embed_dim   = int(os.environ.get("EMBED_DIM", 768))

    # --- embedding backend ---
    # 'st'    = sentence-transformers on the GPU (FAST bulk path; the GTX 1080 is
    #           Pascal/SM6.1, which HuggingFace TEI does NOT support, so we drive
    #           torch directly instead).
    # 'ollama'= zero-setup fallback via the running Ollama (SLOW: ~32 chunks/s).
    embed_backend = os.environ.get("EMBED_BACKEND", "st")    # st | ollama
    embed_model   = os.environ.get("EMBED_MODEL",
                        "nomic-ai/nomic-embed-text-v1.5" if embed_backend == "st"
                        else "nomic-embed-text")
    # 256 OOMs on the 8 GB GTX 1080; 32 is fastest for nomic-768 (padding-bound).
    embed_batch   = int(os.environ.get("EMBED_BATCH", 32))
    device        = os.environ.get("DEVICE", "cuda")         # cuda | cpu
    normalize     = _b("NORMALIZE", True)

    # nomic-embed-text REQUIRES task prefixes. Without them retrieval quality drops.
    doc_prefix   = os.environ.get("DOC_PREFIX", "search_document: ")
    query_prefix = os.environ.get("QUERY_PREFIX", "search_query: ")

    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

    # --- chunking ---
    chunk_size    = int(os.environ.get("CHUNK_SIZE", 1200))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 180))

    # --- ingest batching ---
    # Articles buffered per DB transaction (one COPY + cursor bump + commit).
    article_batch = int(os.environ.get("ARTICLE_BATCH", 400))
    # Stop after this many NEW articles this run (0 = unlimited). Use for a
    # subset run to validate quality before committing to the full corpus.
    max_articles = int(os.environ.get("MAX_ARTICLES", 0))

    # --- query / generation ---
    top_k       = int(os.environ.get("TOP_K", 6))
    ef_search   = int(os.environ.get("EF_SEARCH", 100))      # HNSW recall knob
    chat_model  = os.environ.get("CHAT_MODEL", "qwen2.5:14b")
    num_ctx     = int(os.environ.get("NUM_CTX", 8192))
    temperature = float(os.environ.get("TEMPERATURE", 0.2))

    # --- index build ---
    maintenance_work_mem = os.environ.get("MAINTENANCE_WORK_MEM", "16GB")
    parallel_workers     = int(os.environ.get("PARALLEL_WORKERS", 8))
    index_type           = os.environ.get("INDEX_TYPE", "hnsw")   # hnsw | ivfflat
