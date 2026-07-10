// Naive-but-decent chunker: split on blank lines (paragraphs), then pack
// paragraphs into ~maxChars windows so chunks stay semantically whole.
// Swap this for a token-aware or heading-aware splitter later.
export function chunk(text, maxChars = 1200) {
    const paras = text
        .split(/\n\s*\n/)
        .map((p) => p.trim())
        .filter(Boolean);

    const chunks = [];
    let buf = '';
    for (const p of paras) {
        if (buf && buf.length + p.length + 2 > maxChars) {
            chunks.push(buf);
            buf = '';
        }
        buf = buf ? `${buf}\n\n${p}` : p;
        // Hard-split a single oversized paragraph.
        while (buf.length > maxChars) {
            chunks.push(buf.slice(0, maxChars));
            buf = buf.slice(maxChars);
        }
    }
    if (buf) chunks.push(buf);
    return chunks;
}
