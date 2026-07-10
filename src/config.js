// Config from env, with sensible defaults. Run scripts with `node --env-file=.env`
// (npm scripts already do). No dotenv dependency needed on Node >= 20.6.
export const cfg = {
    db: process.env.DATABASE_URL || 'postgres://rag:rag@localhost:5462/rag',
    ollamaUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
    embedModel: process.env.EMBED_MODEL || 'nomic-embed-text',
    embedDim: Number(process.env.EMBED_DIM || 768),
    chatModel: process.env.CHAT_MODEL || 'llama3.1:8b',
    topK: Number(process.env.TOP_K || 5),
    // Generation context window. Must be large enough to hold the retrieved
    // passages + the question, or Ollama truncates the prompt and the model
    // "loses" the question and refuses. 8k comfortably fits TOP_K=5.
    numCtx: Number(process.env.NUM_CTX || 8192),
    // Lower = more grounded, less creative.
    temperature: Number(process.env.TEMPERATURE ?? 0.2)
};
