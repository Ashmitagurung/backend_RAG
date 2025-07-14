from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        self._sentence_transformer = None
        self._model_name = 'all-MiniLM-L6-v2'  # 384 dimensions
        
    def get_sentence_transformer(self):
        """Lazy load sentence transformer"""
        if self._sentence_transformer is None:
            try:
                logger.info(f"Loading sentence transformer model: {self._model_name}")
                self._sentence_transformer = SentenceTransformer(self._model_name)
                logger.info("✅ Sentence transformer model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load sentence transformer: {e}")
                raise
        return self._sentence_transformer

    def generate_sentence_transformer_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict]:
        """Generate embeddings using sentence transformers with metrics"""
        if not texts:
            return [], {"error": "No texts provided"}
            
        start_time = time.time()

        try:
            model = self.get_sentence_transformer()
            embeddings = model.encode(
                texts, 
                show_progress_bar=len(texts) > 10,
                batch_size=32,
                normalize_embeddings=True
            )
            
            processing_time = time.time() - start_time
            
            metrics = {
                "model": self._model_name,
                "total_texts": len(texts),
                "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "processing_time": processing_time,
                "avg_text_length": sum(len(text) for text in texts) / len(texts) if texts else 0,
                "status": "success"
            }
            
            return embeddings.tolist(), metrics
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return [], {"error": str(e), "status": "failed"}

    def generate_embeddings(self, texts: List[str], model: str = "sentence-transformer") -> Tuple[List[List[float]], Dict]:
        """Main embedding generation method with metrics"""
        if model == "sentence-transformer":
            return self.generate_sentence_transformer_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

# Global instance
embedding_generator = EmbeddingGenerator()