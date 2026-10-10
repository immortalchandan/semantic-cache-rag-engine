import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import QueryRequest, QueryResponse, IngestDocument
from app.embedding import get_embedding_service
from app.semantic_cache import SemanticCache
from app.retriever import HybridRetriever
from app.llm import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Core Singletons
semantic_cache = SemanticCache(default_threshold=0.85)
hybrid_retriever = HybridRetriever(rrf_k=60)
llm_service = LLMService()


# Pre-seed sample enterprise documents for instant demo showcase
SAMPLE_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Payment & Refund Policy",
        "content": "Customers are eligible for a 100% full refund within 30 days of purchase. Refunds are processed back to the original payment method within 3 to 5 business days without cancellation penalties.",
        "metadata": {"category": "billing"}
    },
    {
        "id": "doc-002",
        "title": "API Rate Limiting & Quotas",
        "content": "Production REST API endpoints are limited to 2,000 requests per minute per IP using a token bucket algorithm. Exceeding limits results in HTTP 429 Too Many Requests with a Retry-After header.",
        "metadata": {"category": "infrastructure"}
    },
    {
        "id": "doc-003",
        "title": "Microservices Security Architecture",
        "content": "All inter-service microservice communications require mutual TLS (mTLS) with Ed25519 certificates rotated every 90 days. User sessions utilize stateless JWT tokens signed with RS256.",
        "metadata": {"category": "security"}
    },
    {
        "id": "doc-004",
        "title": "Database Read Replicas & High Availability",
        "content": "PostgreSQL read replicas operate with asynchronous streaming replication. Read queries are routed via HAProxy to achieve under 5ms replica lag and 99.99% system availability.",
        "metadata": {"category": "database"}
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Semantic Caching RAG Engine...")
    # Pre-load embedding model and ingest knowledge base
    get_embedding_service()
    hybrid_retriever.ingest_documents(SAMPLE_DOCUMENTS)
    logger.info("Engine startup complete! Ingested %d default knowledge chunks.", len(SAMPLE_DOCUMENTS))
    yield


app = FastAPI(
    title="AuraRAG — Semantic Caching & Hybrid Retrieval Engine",
    description="Sub-15ms vector cache paired with dense-sparse hybrid search and reciprocal rank fusion.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")


@app.post("/v1/query", response_model=QueryResponse)
async def query_pipeline(req: QueryRequest):
    start_time = time.perf_counter()
    embedding_service = get_embedding_service()

    # Step 1: Compute query vector embedding (normalized 384-d vector)
    query_vector = embedding_service.embed_text(req.query)

    # Step 2: Check Semantic Vector Cache (sub-15ms cosine similarity lookup)
    cache_hit, cached_response, similarity_score = semantic_cache.get_cached(
        query_vector, threshold=req.similarity_threshold
    )

    if cache_hit:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            query=req.query,
            response=cached_response,
            cache_hit=True,
            similarity_score=round(similarity_score, 4) if similarity_score else None,
            latency_ms=round(latency_ms, 2),
            retrieval_method="SEMANTIC_CACHE",
            retrieved_chunks=[]
        )

    # Step 3: Cache Miss -> Run Hybrid Dense + Sparse Retrieval with RRF
    retrieved_chunks = hybrid_retriever.retrieve(req.query, query_vector, top_k=req.top_k)

    # Step 4: Context Assembly & Generation
    llm_response = await llm_service.generate_response(req.query, retrieved_chunks)

    # Step 5: Store query & response in Semantic Cache for future queries
    semantic_cache.store(req.query, query_vector, llm_response)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return QueryResponse(
        query=req.query,
        response=llm_response,
        cache_hit=False,
        similarity_score=round(similarity_score, 4) if similarity_score else None,
        latency_ms=round(latency_ms, 2),
        retrieval_method="HYBRID_RRF_RERANK",
        retrieved_chunks=retrieved_chunks
    )


@app.post("/v1/documents", status_code=201)
def ingest_documents_endpoint(docs: List[IngestDocument]):
    records = [d.model_dump() for d in docs]
    hybrid_retriever.ingest_documents(records)
    return {"status": "success", "indexed_count": len(docs), "total_documents": len(hybrid_retriever.documents)}


@app.delete("/v1/cache", status_code=200)
def clear_semantic_cache():
    semantic_cache.clear()
    return {"status": "success", "message": "Semantic cache cleared"}


@app.get("/v1/stats")
def get_engine_stats():
    return {
        "cached_queries_count": semantic_cache.count(),
        "indexed_documents_count": len(hybrid_retriever.documents),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "similarity_metric": "cosine"
    }
