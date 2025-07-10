from typing import List
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings

class EmbeddingGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        for text in texts:
            response = self.openai_client.embeddings.create(
                input=text,
                model=settings.EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers"""
        embeddings = self.sentence_transformer.encode(texts)
        return embeddings.tolist()
    
    def generate_embeddings(self, texts: List[str], model: str = "openai") -> List[List[float]]:
        """Main embedding generation method"""
        if model == "openai":
            return self.generate_openai_embeddings(texts)
        elif model == "sentence-transformer":
            return self.generate_sentence_transformer_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding model: {model}")

embedding_generator = EmbeddingGenerator()