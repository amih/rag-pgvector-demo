import { cfg } from './config.js';

// A single (non-streamed) chat completion from a local Ollama model.
// Uses /api/chat with a system + user message — better instruction-following
// than a single raw prompt — and pins num_ctx so the prompt is never truncated.
export async function chat(system, user) {
    const res = await fetch(`${cfg.ollamaUrl}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
            model: cfg.chatModel,
            stream: false,
            options: { temperature: cfg.temperature, num_ctx: cfg.numCtx },
            messages: [
                { role: 'system', content: system },
                { role: 'user', content: user }
            ]
        })
    });
    if (!res.ok) {
        throw new Error(`Ollama chat HTTP ${res.status}: ${await res.text()}`);
    }
    const json = await res.json();
    return (json.message?.content || '').trim();
}
