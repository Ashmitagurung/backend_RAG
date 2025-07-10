from typing import List, Dict, Tuple
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings
import time

class EmbeddingGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self._sentence_transformer = None
    
    def get_sentence_transformer(self):
        """Lazy load sentence transformer"""
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_transformer
    
    def generate_openai_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict]:
        """Generate embeddings using OpenAI API with metrics"""
        start_time = time.time()
        embeddings = []
        
        # Batch processing for efficiency
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                input=batch,
                model=settings.EMBEDDING_MODEL
            )
            
            for data in response.data:
                embeddings.append(data.embedding)
        
        processing_time = time.time() - start_time
        
        metrics = {
            "model": "openai",
            "total_texts": len(texts),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "processing_time": processing_time,
            "tokens_used": sum(len(text.split()) for text in texts)
        }
        
        return embeddings, metrics
    
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
    
    def generate_embeddings(self, texts: List[str], model: str = "openai") -> Tuple[List[List[float]], Dict]:
        """Main embedding generation method with metrics"""
        if model == "openai":
            return self.generate_openai_embeddings(texts)
        elif model == "sentence-transformer":
            return self.generate_sentence_transformer_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

embedding_generator = EmbeddingGenerator()