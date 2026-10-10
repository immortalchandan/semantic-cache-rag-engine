from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    query: str = Field(..., description="User prompt or query string")
    similarity_threshold: float = Field(0.85, ge=0.5, le=1.0, description="Cosine similarity threshold for semantic cache hits")
    top_k: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")
    stream: bool = Field(False, description="Whether to stream response tokens")
    api_key: Optional[str] = Field(None, description="Optional user-provided Gemini API key")


class IngestDocument(BaseModel):
    id: Optional[str] = None
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    response: str
    cache_hit: bool
    similarity_score: Optional[float] = None
    matched_cached_query: Optional[str] = None
    latency_ms: float
    retrieval_method: str  # "SEMANTIC_CACHE" or "HYBRID_RRF_RERANK"
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
