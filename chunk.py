"""
Text chunking for RAG: sliding window by character.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks. Empty or whitespace-only chunks are excluded.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if chunk_size <= 0:
        return [text] if text else []
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks
