import uuid
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from app.models import RetrievedChunk
from app.embedding import get_embedding_service
from app.chunker import chunk_text

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Production-grade Hybrid Dense-Sparse Retriever.
    Combines:
    1. Dense Vector Search (cosine similarity over 384-d embeddings)
    2. Sparse BM25 Keyword Search (token frequency matching for exact terms)
    Fuses rankings using Reciprocal Rank Fusion (RRF, k=60).
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.documents: List[Dict[str, Any]] = []
        self.doc_vectors: List[np.ndarray] = []
        self.bm25_index: BM25Okapi = None
        self.tokenized_corpus: List[List[str]] = []

    def ingest_documents(self, docs: List[Dict[str, Any]]):
        embedding_service = get_embedding_service()
        chunked_records = []

        for d in docs:
            raw_content = d.get("content", "")
            title = d.get("title", "Untitled")
            parent_id = d.get("id") or str(uuid.uuid4())
            metadata = d.get("metadata", {})

            # Split document into concise 2-4 sentence chunks
            chunks = chunk_text(raw_content, max_chunk_sentences=3, overlap_sentences=1)
            for idx, c in enumerate(chunks):
                chunk_record = {
                    "id": f"{parent_id}_chunk_{idx}",
                    "title": title,
                    "content": c,
                    "metadata": {**metadata, "parent_id": parent_id, "chunk_index": idx}
                }
                chunked_records.append(chunk_record)

        if not chunked_records:
            return

        new_texts = [r["content"] for r in chunked_records]
        new_vectors = embedding_service.embed_batch(new_texts)

        for record, vec in zip(chunked_records, new_vectors):
            self.documents.append(record)
            self.doc_vectors.append(vec)
            self.tokenized_corpus.append(record["content"].lower().split())

        # Re-index BM25
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

        logger.info("Ingested %d raw documents into %d semantic chunks. Total index size: %d",
                    len(docs), len(chunked_records), len(self.documents))

    def retrieve(self, query: str, query_vector: np.ndarray, top_k: int = 3) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        # 1. Dense Search
        dense_scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            sim = float(np.dot(query_vector, doc_vec))
            dense_scores.append((idx, sim))
        dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)

        # 2. Sparse BM25 Search
        query_tokens = query.lower().split()
        bm25_raw_scores = self.bm25_index.get_scores(query_tokens) if self.bm25_index else [0.0] * len(self.documents)
        bm25_scores = [(idx, float(score)) for idx, score in enumerate(bm25_raw_scores)]
        sparse_ranked = sorted(bm25_scores, key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        # score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(dense_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (doc_idx, _) in enumerate(sparse_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort by composite RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in fused:
            doc = self.documents[doc_idx]
            results.append(RetrievedChunk(
                id=doc["id"],
                text=doc["content"],
                score=round(score, 5),
                source=doc["title"],
                metadata=doc["metadata"]
            ))

        return results
