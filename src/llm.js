import { cfg } from './config.js';

// A single (non-streamed) completion from a local Ollama model.
export async function generate(prompt) {
    const res = await fetch(`${cfg.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: cfg.chatModel, prompt, stream: false })
    });
    if (!res.ok) {
        throw new Error(`Ollama generate HTTP ${res.status}: ${await res.text()}`);
    }
    const json = await res.json();
    return (json.response || '').trim();
}
