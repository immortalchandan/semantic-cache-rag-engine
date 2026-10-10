# AuraRAG — Low-Latency Semantic Caching & Hybrid Retrieval Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg?style=flat&logo=render)](https://semantic-cache-rag-engine.onrender.com/)

An enterprise-grade, high-throughput AI engineering engine combining **sub-15ms semantic vector caching** with a **hybrid dense-sparse retrieval pipeline** (BM25 + Dense Embeddings fused via Reciprocal Rank Fusion) and automated **Google Gemini Flash** multi-model generation.

🎮 **Live Interactive Telemetry Console:** [https://semantic-cache-rag-engine.onrender.com/](https://semantic-cache-rag-engine.onrender.com/)

---

## 🏗️ System Architecture & Retrieval Pipeline

```mermaid
flowchart TD
    User(["🌐 Client Query / Prompt"]) -->|"POST /v1/query"| Gateway["⚡ FastAPI Async Gateway<br/>(Token Bucket Rate Limiter: 20 req/min)"]
    
    Gateway -->|"Mean-Pooled L2 Normalization"| Embedding["🧠 all-MiniLM-L6-v2<br/>(384-Dimensional Dense Vector)"]
    
    Embedding -->|"Cosine Similarity Check"| Cache{"⚡ Semantic Vector Cache<br/>(Cosine Threshold: 0.75 - 0.85)"}
    
    subgraph S1 ["Sub-15ms Semantic Cache Fast-Path"]
        Cache -->|"Cosine Sim >= 0.75 (HIT)"| CacheHit["🚀 Instant Vector Return<br/>(Latency: 5 - 15ms | Bypasses LLM)"]
    end
    
    subgraph S2 ["Hybrid Dense-Sparse RAG Path (Cache MISS)"]
        Cache -->|"Cosine Sim < 0.75 (MISS)"| HybridRetriever["🔍 Hybrid Retrieval Engine"]
        
        HybridRetriever -->|"Dense Cosine Search"| DenseRank["📊 Dense Vector Ranking<br/>(Semantic Concepts)"]
        HybridRetriever -->|"Sparse Keyword Matching"| SparseRank["🔤 BM25 Lexical Ranking<br/>(Exact Tokens & Acronyms)"]
        
        DenseRank -->|"Composite Rank Scoring"| RRF["⚖️ Reciprocal Rank Fusion (RRF, k=60)<br/>score = 1/(60 + r_dense) + 1/(60 + r_sparse)"]
        SparseRank -->|"Composite Rank Scoring"| RRF
        
        RRF -->|"Top-k Re-ranked Context"| LLM["✨ Google Gemini Flash Ladder<br/>(gemini-3.5-flash / gemini-3.5-flash-lite)"]
        LLM -->|"Token-Budgeted 2-Sentence Summary"| StoreCache["💾 Store Vector & Response in Cache"]
    end
    
    StoreCache --> FinalResp(["📤 Client HTTP 200 OK Response"])
    CacheHit --> FinalResp

    classDef primary fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef storage fill:#181825,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef success fill:#313244,stroke:#a6e3a1,stroke-width:2px,color:#a6e3a1;
    classDef llm fill:#313244,stroke:#f9e2af,stroke-width:2px,color:#f9e2af;

    class Gateway,Embedding,HybridRetriever,RRF primary;
    class Cache,DenseRank,SparseRank,StoreCache storage;
    class CacheHit success;
    class LLM llm;
```

---

## 🚀 Key Technical Highlights

1. **Sub-15ms Semantic Vector Caching:**
   - Unlike naive string caches, semantic caching uses cosine similarity across dense embedding vectors.
   - Saves 70%+ of LLM API costs and reduces query response latency from ~1,500ms to **under 10ms**.
2. **Hybrid Dense + Sparse Search (RRF):**
   - **Dense Vectors:** Understand broad semantic intent and concepts.
   - **BM25 Sparse Tokens:** Capture exact acronyms, error codes, and part numbers.
   - Combined via Reciprocal Rank Fusion:
     $$\text{RRF Score} = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
3. **Interactive Visual Dashboard:**
   - Real-time telemetry tracking: **Total Queries**, **Cache Hits**, **Average Latency**, and **Token Cost Savings**.

---

## 💻 Local Setup & Execution

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/immortalchandan/semantic-cache-rag-engine.git
cd semantic-cache-rag-engine
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive console at:
👉 **`http://localhost:8000`**

### 3. Run Tests
```bash
pytest tests
```
