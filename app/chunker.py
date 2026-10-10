import re
from typing import List, Dict, Any


def chunk_text(text: str, max_chunk_sentences: int = 3, overlap_sentences: int = 1) -> List[str]:
    """
    Sentence-based semantic chunker with sliding window overlap.
    Breaks large documents down into 2-4 sentence cohesive chunks.
    """
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    if len(sentences) <= max_chunk_sentences:
        return [" ".join(sentences)]

    chunks = []
    step = max(1, max_chunk_sentences - overlap_sentences)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + max_chunk_sentences])
        chunks.append(chunk)
        if i + max_chunk_sentences >= len(sentences):
            break

    return chunks
