# wikirag/chunker.py
"""Split document text into overlapping fixed-size chunks."""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters,
    with overlap characters carried over between consecutive chunks.
    Splits at word boundaries.
    """
    if not text or not text.strip():
        return []

    text = " ".join(text.split())
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        boundary = text.rfind(" ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        next_start = boundary - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_document(
    title: str,
    entity_type: str,
    text: str,
    url: str = "",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[dict]:
    """
    Chunk a document and attach metadata.
    chunk_size and overlap are passed in directly from the active ChunkProfile.
    """
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "text": chunk,
            "title": title,
            "entity_type": entity_type,
            "chunk_index": i,
            "url": url,
        }
        for i, chunk in enumerate(raw_chunks)
    ]
