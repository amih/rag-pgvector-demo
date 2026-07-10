// Embed the question, retrieve nearest chunks by cosine distance, answer from them.
import { pool, toVector } from './db.js';
import { embed } from './embed.js';
import { chat } from './llm.js';
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

// 2. Augment: build a grounded prompt. The question goes FIRST (so it can never
//    be truncated away), the passages follow. The system message keeps the model
//    helpful and synthesizing rather than refusing unless the passages are unrelated.
const context = rows.map((r, i) => `[${i + 1}] (${r.source})\n${r.content}`).join('\n\n');

const system =
    "You answer questions about Ami Heines's writing on cloud and AI cost control. " +
    'Use the provided context passages as your source of truth and synthesize across them. ' +
    'Give a direct, practical answer — short steps or bullet points are good. ' +
    'If the passages only partially cover the question, answer with what they do say instead of refusing. ' +
    'Only say you cannot answer if the passages are genuinely unrelated to the question. ' +
    'Do not invent facts beyond the passages.';

const user = `Question: ${question}\n\nContext passages:\n${context}\n\nAnswer the question above using these passages.`;

// 3. Generate.
const answer = await chat(system, user);
console.log(`\nAnswer:\n${answer}\n`);
await pool.end();
