import json
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Sub-15ms Semantic Vector Cache.
    Maintains embeddings of prior user prompts and checks cosine similarity.
    If similarity >= threshold, returns the cached answer without hitting external LLMs.
    """

    def __init__(self, default_threshold: float = 0.85):
        self.default_threshold = default_threshold
        # In-memory storage for high-speed local indexing and fallback:
        # entries: list of dicts: {"query": str, "response": str, "vector": np.ndarray, "metadata": dict}
        self.cache_entries: List[Dict[str, Any]] = []

    def get_cached(self, query_vector: np.ndarray, threshold: Optional[float] = None) -> Tuple[bool, Optional[str], Optional[float]]:
        target_threshold = threshold if threshold is not None else self.default_threshold

        if not self.cache_entries:
            return False, None, None

        # Compute cosine similarity across all cached query vectors
        # Since vectors are normalized (L2 norm = 1), dot product is cosine similarity
        best_score = -1.0
        best_response = None

        for entry in self.cache_entries:
            vector = entry["vector"]
            similarity = float(np.dot(query_vector, vector))
            if similarity > best_score:
                best_score = similarity
                best_response = entry["response"]

        if best_score >= target_threshold:
            logger.info("Semantic cache HIT! Cosine similarity: %.4f >= threshold: %.2f", best_score, target_threshold)
            return True, best_response, best_score

        return False, None, best_score if best_score > 0 else None

    def store(self, query: str, query_vector: np.ndarray, response: str, metadata: Optional[Dict[str, Any]] = None):
        self.cache_entries.append({
            "query": query,
            "response": response,
            "vector": query_vector,
            "metadata": metadata or {}
        })
        logger.info("Stored query into semantic cache. Total cached items: %d", len(self.cache_entries))

    def clear(self):
        self.cache_entries.clear()

    def count(self) -> int:
        return len(self.cache_entries)
