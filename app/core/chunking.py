from typing import List, Dict, Tuple
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self._sentence_model = None
    
    def get_sentence_model(self):
        """Lazy load sentence transformer model"""
        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence model loaded for semantic chunking")
            except Exception as e:
                logger.error(f"❌ Failed to load sentence model: {e}")
                raise
        return self._sentence_model
    
    def recursive_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[str], Dict]:
        """Recursive character-based chunking with metrics"""
        start_time = time.time()
        
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)
            
            processing_time = time.time() - start_time
            
            metrics = {
                "method": "recursive",
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "processing_time": processing_time,
                "parameters": {"chunk_size": chunk_size, "overlap": overlap},
                "status": "success"
            }
            
            logger.info(f"✅ Recursive chunking completed: {len(chunks)} chunks")
            return chunks, metrics
            
        except Exception as e:
            logger.error(f"❌ Recursive chunking failed: {e}")
            return [], {"error": str(e), "status": "failed"}
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> Tuple[List[str], Dict]:
        """Semantic similarity-based chunking with metrics"""
        start_time = time.time()
        
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) < 2:
                return [text], {
                    "method": "semantic",
                    "total_chunks": 1,
                    "avg_chunk_size": len(text),
                    "processing_time": time.time() - start_time,
                    "parameters": {"similarity_threshold": similarity_threshold},
                    "status": "success"
                }
            
            # Generate embeddings
            model = self.get_sentence_model()
            embeddings = model.encode(sentences, show_progress_bar=False)
            
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                similarity = cosine_similarity(
                    embeddings[i-1].reshape(1, -1), 
                    embeddings[i].reshape(1, -1)
                )[0][0]
                
                if similarity > similarity_threshold and len('. '.join(current_chunk)) < 1500:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentences[i]]
            
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            
            processing_time = time.time() - start_time
            
            metrics = {
                "method": "semantic",
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "processing_time": processing_time,
                "parameters": {"similarity_threshold": similarity_threshold},
                "status": "success"
            }
            
            logger.info(f"✅ Semantic chunking completed: {len(chunks)} chunks")
            return chunks, metrics
            
        except Exception as e:
            logger.error(f"❌ Semantic chunking failed: {e}")
            # Fallback to recursive chunking
            return self.recursive_chunking(text)
    
    def custom_chunking(self, text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> Tuple[List[str], Dict]:
        """Custom token-based chunking with smart boundaries"""
        start_time = time.time()
        
        try:
            tokens = self.encoding.encode(text)
            chunks = []
            
            for i in range(0, len(tokens), max_tokens - overlap_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                # Try to end at sentence boundary
                if i + max_tokens < len(tokens):
                    # Look for sentence endings
                    for ending in ['. ', '! ', '? ', '\n\n']:
                        last_ending = chunk_text.rfind(ending)
                        if last_ending > len(chunk_text) * 0.7:
                            chunk_text = chunk_text[:last_ending + len(ending)]
                            break
                
                chunks.append(chunk_text.strip())
            
            processing_time = time.time() - start_time
            
            metrics = {
                "method": "custom",
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "processing_time": processing_time,
                "parameters": {"max_tokens": max_tokens, "overlap_tokens": overlap_tokens},
                "status": "success"
            }
            
            logger.info(f"✅ Custom chunking completed: {len(chunks)} chunks")
            return chunks, metrics
            
        except Exception as e:
            logger.error(f"❌ Custom chunking failed: {e}")
            return [], {"error": str(e), "status": "failed"}
    
    def chunk_document(self, text: str, method: str = "recursive") -> Tuple[List[str], Dict]:
        """Main chunking method dispatcher with metrics"""
        if not text.strip():
            return [], {"error": "Empty text provided", "status": "failed"}
            
        if method == "recursive":
            return self.recursive_chunking(text)
        elif method == "semantic":
            return self.semantic_chunking(text)
        elif method == "custom":
            return self.custom_chunking(text)
        else:
            logger.warning(f"Unknown chunking method: {method}, using recursive")
            return self.recursive_chunking(text)

# Global instance
chunker = DocumentChunker()