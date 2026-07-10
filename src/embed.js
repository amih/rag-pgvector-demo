import { cfg } from './config.js';

// One embedding vector for a piece of text, via Ollama's local API.
export async function embed(text) {
    const res = await fetch(`${cfg.ollamaUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: cfg.embedModel, prompt: text })
    });
    if (!res.ok) {
        throw new Error(`Ollama embeddings HTTP ${res.status}: ${await res.text()}`);
    }
    const json = await res.json();
    if (!Array.isArray(json.embedding)) {
        throw new Error(`No embedding in Ollama response (is "${cfg.embedModel}" pulled?)`);
    }
    return json.embedding;
}
