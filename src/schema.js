// Create the extension, table, and ANN index. Idempotent.
import { pool } from './db.js';
import { cfg } from './config.js';

await pool.query('CREATE EXTENSION IF NOT EXISTS vector');

await pool.query(`
    CREATE TABLE IF NOT EXISTS chunks (
        id           bigserial PRIMARY KEY,
        source       text NOT NULL,
        chunk_index  int  NOT NULL,
        content      text NOT NULL,
        embedding    vector(${cfg.embedDim}) NOT NULL
    )
`);

// HNSW index for approximate nearest-neighbour search by cosine distance.
await pool.query(`
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
`);

console.log(`schema applied (embedding dim = ${cfg.embedDim}).`);
await pool.end();
