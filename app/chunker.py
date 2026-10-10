import re
from typing import List


def chunk_text(text: str, target_chunk_size: int = 400, overlap_size: int = 80) -> List[str]:
    """
    Paragraph and sentence-aware chunker that preserves complete ideas and sentences.
    Ensures chunks are cohesive, self-contained, and typically 3-5 complete sentences (~300-500 chars).
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # If document is already reasonably short, keep it as a complete entity
    if len(clean_text) <= target_chunk_size + 100:
        return [clean_text]

    # Split into clean paragraphs first
    paragraphs = [p.strip() for p in re.split(r'\n+', clean_text) if p.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        # Split paragraph into complete sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > target_chunk_size and current_chunk:
                # Store completed chunk
                chunks.append(" ".join(current_chunk))
                # Keep last sentence as overlap for conversational continuity
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], sent]
                    current_len = len(current_chunk[0]) + sent_len + 1
                else:
                    current_chunk = [sent]
                    current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
