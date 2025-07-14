from typing import List, Dict, Tuple, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
import uuid
import time
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is required")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if settings.PINECONE_INDEX_NAME not in index_names:
                logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(10)
            
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info("✅ Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            raise
    
    def test_connection(self):
        """Test Pinecone connection"""
        try:
            if self.index:
                stats = self.index.describe_index_stats()
                logger.info(f"Pinecone connection test successful. Index stats: {stats}")
                return True
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            raise
        return False
    
    def store_embeddings(self, embeddings: List[List[float]], 
                        texts: List[str], 
                        metadata: List[Dict]) -> List[str]:
        """Store embeddings in vector database"""
        if not embeddings or not texts:
            return []
            
        try:
            vectors = []
            ids = []
            
            for i, (embedding, text, meta) in enumerate(zip(embeddings, texts, metadata)):
                vector_id = str(uuid.uuid4())
                ids.append(vector_id)
                
                # Limit metadata size for Pinecone
                limited_meta = {
                    "document_id": meta.get("document_id", ""),
                    "filename": meta.get("filename", "")[:100],
                    "chunk_index": meta.get("chunk_index", 0),
                    "chunking_method": meta.get("chunking_method", ""),
                    "embedding_model": meta.get("embedding_model", ""),
                    "text": text[:1000]  # Limit text size
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": limited_meta
                })
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info(f"✅ Stored {len(vectors)} embeddings successfully")
            return ids
            
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            raise
    
    def similarity_search_cosine(self, query_embedding: List[float], 
                               top_k: int = 5, 
                               filter_dict: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """Cosine similarity search with metrics"""
        start_time = time.time()
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            search_time = time.time() - start_time
            
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata
                })
            
            metrics = {
                "search_method": "cosine",
                "search_time": search_time,
                "results_count": len(formatted_results),
                "top_k": top_k,
                "status": "success"
            }
            
            return formatted_results, metrics
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return [], {"error": str(e), "status": "failed"}
    
    def similarity_search_euclidean(self, query_embedding: List[float], 
                                  top_k: int = 5, 
                                  filter_dict: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """Euclidean similarity search simulation"""
        # For demonstration - transform cosine scores
        results, metrics = self.similarity_search_cosine(query_embedding, top_k * 2, filter_dict)
        
        # Transform scores to simulate euclidean distance
        for result in results:
            result["score"] = 2 - result["score"]  # Transform cosine to euclidean-like
        
        # Sort by new scores and take top_k
        results.sort(key=lambda x: x["score"])
        results = results[:top_k]
        
        metrics["search_method"] = "euclidean"
        return results, metrics
    
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
        try:
            self.index.delete(filter={"document_id": document_id})
            logger.info(f"✅ Deleted vectors for document: {document_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete vectors for document {document_id}: {e}")
            raise

# Global instance
vector_store = VectorStore()