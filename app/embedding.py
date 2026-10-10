import logging
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Production embedding service utilizing direct HuggingFace Transformers
    with mean pooling and L2 normalization over 'sentence-transformers/all-MiniLM-L6-v2'.
    Bypasses scikit-learn/scipy C-extensions to guarantee 100% portability.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Loading tokenizer and transformer model: %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.dimension = 384
        self._initialized = True
        logger.info("EmbeddingService initialized successfully.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings for cosine similarity
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().astype(np.float32)


_service_instance = None


def get_embedding_service() -> EmbeddingService:
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance
