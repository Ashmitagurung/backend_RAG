from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import time

class EmbeddingGenerator:
    def __init__(self):
        self._sentence_transformer = None

    def get_sentence_transformer(self):
        """Lazy load sentence transformer"""
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_transformer

    def generate_sentence_transformer_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict]:
        """Generate embeddings using sentence transformers with metrics"""
        start_time = time.time()

        model = self.get_sentence_transformer()
        embeddings = model.encode(texts, show_progress_bar=True)

        processing_time = time.time() - start_time

        metrics = {
            "model": "sentence-transformer",
            "total_texts": len(texts),
            "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "processing_time": processing_time,
            "avg_text_length": sum(len(text) for text in texts) / len(texts) if texts else 0
        }

        return embeddings.tolist(), metrics

    def generate_embeddings(self, texts: List[str], model: str = "sentence-transformer") -> Tuple[List[List[float]], Dict]:
        """Main embedding generation method with metrics"""
        if model == "sentence-transformer":
            return self.generate_sentence_transformer_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

embedding_generator = EmbeddingGenerator()
