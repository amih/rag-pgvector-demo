// Embed the question, retrieve nearest chunks by cosine distance, answer from them.
import { pool, toVector } from './db.js';
import { embed } from './embed.js';
import { generate } from './llm.js';
import { cfg } from './config.js';

const question = process.argv.slice(2).join(' ').trim();
if (!question) {
    console.error('Usage: npm run query -- "your question"');
    process.exit(1);
}

// 1. Retrieve: nearest chunks to the question embedding.
//    `<=>` is pgvector's cosine distance; similarity = 1 - distance.
const qvec = toVector(await embed(question));
const { rows } = await pool.query(
    `SELECT source, content, 1 - (embedding <=> $1::vector) AS similarity
       FROM chunks
       ORDER BY embedding <=> $1::vector
       LIMIT $2`,
    [qvec, cfg.topK]
);

if (rows.length === 0) {
    console.error('No chunks in the DB. Run `npm run ingest` first.');
    process.exit(1);
}

console.log('\nRetrieved context:');
for (const r of rows) console.log(`  [${r.similarity.toFixed(3)}] ${r.source}`);

// 2. Augment: stuff the retrieved chunks into a grounded prompt.
const context = rows.map((r, i) => `[${i + 1}] (${r.source})\n${r.content}`).join('\n\n');
const prompt = [
    'Answer the question using ONLY the context below.',
    'If the answer is not in the context, say you do not know.',
    '',
    'Context:',
    context,
    '',
    `Question: ${question}`,
    '',
    'Answer:'
].join('\n');

// 3. Generate.
const answer = await generate(prompt);
console.log(`\nAnswer:\n${answer}\n`);
await pool.end();
