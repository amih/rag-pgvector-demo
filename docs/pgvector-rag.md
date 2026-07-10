# RAG on Postgres with pgvector

Retrieval-Augmented Generation (RAG) gives a language model facts it was not trained on. Instead of
fine-tuning, you retrieve relevant passages at query time and paste them into the prompt. The model answers
from that supplied context, so answers stay current and grounded, and you can cite the source.

## You do not need a dedicated vector database

A vector search is just: turn text into a list of numbers (an embedding), then find the stored vectors closest
to your query vector. "Closest" is usually cosine distance. Postgres can do this with the `pgvector`
extension — the same database you already run for everything else. No new service, no new backup story, no new
access-control model.

pgvector adds a `vector` column type and distance operators. `<=>` is cosine distance, `<->` is L2 distance,
and `<#>` is negative inner product. You put an approximate-nearest-neighbour index (HNSW or IVFFlat) on the
column so search stays fast as the table grows.

## The three steps of RAG

1. Ingest: split documents into chunks, embed each chunk, store the text and its vector.
2. Retrieve: embed the user's question and select the nearest chunks with an `ORDER BY embedding <=> query`.
3. Generate: put the retrieved chunks into the prompt and ask the model to answer using only that context.

## When to reach for a specialised vector store

pgvector is the right default for most workloads up to millions of vectors. Consider a dedicated store such as
Qdrant, Weaviate, or Milvus only when you need very high vector counts, heavy metadata filtering at scale, or
specialised index types. Start with Postgres; move only when a real limit forces you to.
