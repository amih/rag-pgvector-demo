// Read every .md/.txt in docs/, chunk it, embed each chunk, store in Postgres.
import { readdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, extname } from 'node:path';
import { pool, toVector } from './db.js';
import { embed } from './embed.js';
import { chunk } from './chunk.js';

const here = dirname(fileURLToPath(import.meta.url));
const docsDir = join(here, '..', 'docs');

const files = (await readdir(docsDir)).filter((f) => ['.md', '.txt'].includes(extname(f).toLowerCase()));
if (files.length === 0) {
    console.error(`No .md/.txt files in ${docsDir}. Drop some docs there and re-run.`);
    process.exit(1);
}

// Demo-simple: reload from scratch each run. Delete this to append instead.
await pool.query('TRUNCATE chunks RESTART IDENTITY');

let total = 0;
for (const file of files) {
    const text = await readFile(join(docsDir, file), 'utf8');
    const parts = chunk(text);
    for (let i = 0; i < parts.length; i++) {
        const vec = await embed(parts[i]);
        await pool.query(
            'INSERT INTO chunks (source, chunk_index, content, embedding) VALUES ($1, $2, $3, $4::vector)',
            [file, i, parts[i], toVector(vec)]
        );
        total++;
    }
    console.log(`  ${file}: ${parts.length} chunks`);
}

console.log(`ingested ${total} chunks from ${files.length} file(s).`);
await pool.end();
