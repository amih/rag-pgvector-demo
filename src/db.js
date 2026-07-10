import pg from 'pg';
import { cfg } from './config.js';

export const pool = new pg.Pool({ connectionString: cfg.db });

// pgvector accepts a plain string literal like "[0.1,0.2,0.3]" cast to ::vector.
export const toVector = (arr) => `[${arr.join(',')}]`;
