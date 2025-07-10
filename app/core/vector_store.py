from typing import List, Dict, Tuple
import pinecone
from app.config import settings
import uuid

class VectorStore:
    def __init__(self):
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
                    "text": text
                }
            })
        
        # Batch upsert
        self.index.upsert(vectors)
        return ids
    
    def similarity_search(self, query_embedding: List[float], 
                         top_k: int = 5, 
                         filter_dict: Dict = None) -> List[Dict]:
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        return [{
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "metadata": match.metadata
        } for match in results.matches]
    
    def delete_by_document(self, document_id: str):
        """Delete all vectors for a specific document"""
        self.index.delete(filter={"document_id": document_id})

vector_store = VectorStore()