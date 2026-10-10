import pytest
import numpy as np
from app.semantic_cache import SemanticCache
from app.retriever import HybridRetriever


def test_semantic_cache_exact_and_cosine_hit():
    cache = SemanticCache(default_threshold=0.85)

    # Normalized 384-d vector mock
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0

    cache.store("What is refund policy?", v1, "100% refund in 30 days.")

    # 1. Exact match (cosine = 1.0)
    hit, resp, score = cache.get_cached(v1, threshold=0.85)
    assert hit is True
    assert resp == "100% refund in 30 days."
    assert score == pytest.approx(1.0, 0.001)

    # 2. Near-semantic match (cosine ~ 0.96)
    v2 = np.zeros(384, dtype=np.float32)
    v2[0] = 0.96
    v2[1] = np.sqrt(1 - 0.96**2)  # Maintain unit norm

    hit, resp, score = cache.get_cached(v2, threshold=0.85)
    assert hit is True
    assert resp == "100% refund in 30 days."
    assert score >= 0.85

    # 3. Unrelated query (cosine = 0.0)
    v3 = np.zeros(384, dtype=np.float32)
    v3[2] = 1.0

    hit, resp, score = cache.get_cached(v3, threshold=0.85)
    assert hit is False
    assert resp is None


def test_hybrid_retriever_rrf_scoring():
    retriever = HybridRetriever(rrf_k=60)
    docs = [
        {"id": "1", "title": "Refund", "content": "Customers receive full refund within 30 days."},
        {"id": "2", "title": "API Limit", "content": "Rate limit is 2000 requests per minute."}
    ]

    retriever.ingest_documents(docs)
    assert len(retriever.documents) == 2

    # Query matching doc 1
    query = "How do I get my money back with refund?"
    q_vec = retriever.doc_vectors[0]  # Match doc 1

    chunks = retriever.retrieve(query, q_vec, top_k=1)
    assert len(chunks) == 1
    assert chunks[0].id == "1"
    assert chunks[0].source == "Refund"
