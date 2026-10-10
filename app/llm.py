import os
import re
import logging
from typing import List
import httpx

from app.models import RetrievedChunk
from app.embedding import get_embedding_service

logger = logging.getLogger(__name__)


class LLMService:
    """
    RAG generation engine. Supports external providers (Groq/Gemini) if keys are provided.
    Includes an intelligent extractive sentence ranker and synthesizer that analyzes the retrieved
    document chunks and constructs a crisp, cohesive 2-4 sentence summary tailored directly to the query.
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    async def generate_response(self, query: str, context_chunks: List[RetrievedChunk], user_gemini_key: str = None) -> str:
        # Dynamically read from user request, os.getenv, or saved key
        gemini_key = user_gemini_key or os.getenv("GEMINI_API_KEY") or self.gemini_api_key
        groq_key = os.getenv("GROQ_API_KEY") or self.groq_api_key

        # 1. If Gemini API key is present, use Google Gemini Flash
        if gemini_key:
            try:
                return await self._call_gemini(query, context_chunks, gemini_key)
            except Exception as e:
                logger.warning("Gemini call failed, falling back: %s", e)

        # 2. If Groq API key is present, use ultra-fast Llama 3 on Groq
        if groq_key:
            try:
                return await self._call_groq(query, context_chunks, groq_key)
            except Exception as e:
                logger.warning("Groq call failed, falling back to intelligent local synthesizer: %s", e)

        # 3. Intelligent Extractive Synthesizer: analyzes all chunks, scores each sentence against the query,
        # and composes a fluent, complete 2-3 line response.
        return self._synthesize_locally(query, context_chunks)

    async def _call_gemini(self, query: str, context_chunks: List[RetrievedChunk], api_key: str) -> str:
        if context_chunks:
            context_text = "\n\n".join([f"[{c.source}]: {c.text}" for c in context_chunks])
            prompt = (
                "You are a helpful, precise technical assistant.\n"
                "Analyze the context below to answer the user's query.\n"
                "Requirements:\n"
                "1. Synthesize your own clean, fluent answer strictly based on the context.\n"
                "2. Keep the answer between 2 to 4 complete sentences.\n"
                "3. Do NOT quote long blocks of text verbatim; provide a direct, coherent, standalone response.\n\n"
                f"Context:\n{context_text}\n\n"
                f"User Query: {query}\n\n"
                "Direct Answer (2-4 sentences):"
            )
        else:
            # Query didn't match indexed domain documents; answer directly as an AI assistant
            prompt = (
                "You are a helpful, precise technical assistant.\n"
                "Answer the user query accurately in 2 to 4 concise, complete sentences.\n\n"
                f"User Query: {query}\n\n"
                "Direct Answer (2-4 sentences):"
            )

        # Gemini Flash model fallback ladder: latest Flash models down to earlier versions
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
        }

        async with httpx.AsyncClient(timeout=12.0) as client:
            for model_name in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                            text = candidates[0]["content"]["parts"][0].get("text", "").strip()
                            if text:
                                logger.info("Successfully generated response using model %s: %s", model_name, text[:80])
                                return text
                    else:
                        logger.error("Gemini model %s call failed with HTTP %d: %s", model_name, resp.status_code, resp.text)
                except Exception as e:
                    logger.error("Gemini model %s network/client error: %s", model_name, e)

        # If all Gemini models are exhausted or fail, fall back to intelligent local synthesis
        return self._synthesize_locally(query, context_chunks)

    async def _call_groq(self, query: str, context_chunks: List[RetrievedChunk], api_key: str) -> str:
        context_text = "\n\n".join([f"[{c.source}]: {c.text}" for c in context_chunks])
        prompt = (
            f"You are a helpful, precise technical assistant. "
            f"Answer the user query in 2 to 3 concise, complete, professional sentences using the context below.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 250
                }
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _synthesize_locally(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        if not context_chunks:
            return "I could not find relevant context in the indexed knowledge base to answer this query."

        # Gather sentences from all top retrieved chunks
        all_sentences = []
        sources = []

        for chunk in context_chunks[:3]:
            if chunk.source not in sources:
                sources.append(chunk.source)
            # Split chunk into clean sentences
            raw_sents = re.split(r'(?<=[.!?])\s+', chunk.text.strip())
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) > 20 and clean_s not in all_sentences:
                    all_sentences.append(clean_s)

        if not all_sentences:
            return f"Based on '{context_chunks[0].source}': {context_chunks[0].text}"

        # Analyze sentences against the query vector to pick the 2 most informative sentences
        try:
            embedding_service = get_embedding_service()
            query_vec = embedding_service.embed_text(query)
            sent_vecs = embedding_service.embed_batch(all_sentences)

            scored = []
            for sent, s_vec in zip(all_sentences, sent_vecs):
                import numpy as np
                sim = float(np.dot(query_vec, s_vec))
                scored.append((sent, sim))

            # Sort by relevance to query
            scored.sort(key=lambda x: x[1], reverse=True)
            # Pick top 2 most relevant complete sentences
            selected_sentences = [item[0] for item in scored[:2]]
        except Exception as e:
            logger.warning("Local sentence scoring fallback: %s", e)
            selected_sentences = all_sentences[:2]

        # Ensure every sentence ends cleanly with proper punctuation
        formatted_sents = []
        for s in selected_sentences:
            s_clean = s.strip()
            if not s_clean.endswith(('.', '!', '?')):
                s_clean += '.'
            formatted_sents.append(s_clean)

        combined_text = " ".join(formatted_sents)
        source_label = sources[0] if len(sources) == 1 else f"{sources[0]} and {sources[1]}"
        return f"According to {source_label}, {combined_text}"
