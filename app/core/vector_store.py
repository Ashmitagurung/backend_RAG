from typing import List, Dict, Tuple, Optional
import pinecone
from app.config import settings
import uuid
import time

import os

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY is required")


class VectorStore:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        # Create index if it doesn't exist
        if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                settings.PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
    
    def store_embeddings(self, embeddings: List[List[float]], 
                        texts: List[str], 
                        metadata: List[Dict]) -> List[str]:
        """Store embeddings in vector database"""
        vectors = []
        ids = []
        
        for i, (embedding, text, meta) in enumerate(zip(embeddings, texts, metadata)):
            vector_id = str(uuid.uuid4())
            ids.append(vector_id)
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    **meta,
                    "text": text[:1000],  # Limit text size for metadata
                    "full_text": text
                }
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return ids
    
    def similarity_search_cosine(self, query_embedding: List[float], 
                               top_k: int = 5, 
                               filter_dict: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """Cosine similarity search with metrics"""
        start_time = time.time()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        search_time = time.time() - start_time
        
        formatted_results = [{
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("full_text", match.metadata.get("text", "")),
            "metadata": match.metadata
        } for match in results.matches]
        
        metrics = {
            "search_method": "cosine",
            "search_time": search_time,
            "results_count": len(formatted_results),
            "top_k": top_k
        }
        
        return formatted_results, metrics
    
    def similarity_search_euclidean(self, query_embedding: List[float], 
                                  top_k: int = 5, 
                                  filter_dict: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """Euclidean similarity search (simulated using cosine with normalization)"""
        start_time = time.time()
        
        # For demonstration - Pinecone primarily uses cosine
        # In a real implementation, you might use a different vector DB that supports euclidean
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Get more results to simulate different ranking
            filter=filter_dict,
            include_metadata=True
        )
        
        # Simulate euclidean distance ranking (simplified)
        formatted_results = [{
            "id": match.id,
            "score": 1.0 - (match.score * 0.8),  # Simulate different scoring
            "text": match.metadata.get("full_text", match.metadata.get("text", "")),
            "metadata": match.metadata
        } for match in results.matches]
        
        # Sort by simulated euclidean score and take top_k
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        formatted_results = formatted_results[:top_k]
        
        search_time = time.time() - start_time
        
        metrics = {
            "search_method": "euclidean",
            "search_time": search_time,
            "results_count": len(formatted_results),
            "top_k": top_k
        }
        
        return formatted_results, metrics
    
    def similarity_search(self, query_embedding: List[float], 
                         top_k: int = 5, 
                         filter_dict: Optional[Dict] = None,
                         method: str = "cosine") -> Tuple[List[Dict], Dict]:
        """Main similarity search method"""
        if method == "cosine":
            return self.similarity_search_cosine(query_embedding, top_k, filter_dict)
        elif method == "euclidean":
            return self.similarity_search_euclidean(query_embedding, top_k, filter_dict)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def delete_by_document(self, document_id: str):
        """Delete all vectors for a specific document"""
        self.index.delete(filter={"document_id": document_id})

vector_store = VectorStore()