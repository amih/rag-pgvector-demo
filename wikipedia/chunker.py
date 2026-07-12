"""Overlapping, boundary-aware chunker.

Improvements over the demo's src/chunk.js:
  - fixed-size windows WITH overlap (facts spanning a boundary survive in both),
  - prefer to cut on a sentence ("? "/". ") or word boundary, not mid-word.
nomic-embed-text takes 2048 tokens (~8k chars), so 1200-char chunks are well
within budget; tune CHUNK_SIZE/CHUNK_OVERLAP in .env.
"""


def chunk(text, size=1200, overlap=180):
    text = " ".join(text.split())
    n = len(text)
    if n == 0:
        return []
    if n <= size:
        return [text]

    out = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # look for a sentence break, then a space, in the back half of the window
            floor = start + size // 2
            b = text.rfind(". ", floor, end)
            if b == -1:
                b = text.rfind(" ", floor, end)
            if b != -1:
                end = b + 1
        piece = text[start:end].strip()
        if len(piece) > 40:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out
