from typing import List, Dict
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from langchain.schema import Document
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DocumentChunker:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def recursive_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Recursive character-based chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[str]:
        """Semantic similarity-based chunking"""
        # Split into sentences first
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return [text]
        
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1), 
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity > similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # Start new chunk
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        # Add final chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def custom_chunking(self, text: str, max_tokens: int = 512) -> List[str]:
        """Custom token-based chunking with smart boundaries"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Try to end at sentence boundary
            if i + max_tokens < len(tokens):
                last_period = chunk_text.rfind('.')
                if last_period > len(chunk_text) * 0.8:  # If period is in last 20%
                    chunk_text = chunk_text[:last_period + 1]
            
            chunks.append(chunk_text)
        
        return chunks
    
    def chunk_document(self, text: str, method: str = "recursive") -> List[str]:
        """Main chunking method dispatcher"""
        if method == "recursive":
            return self.recursive_chunking(text)
        elif method == "semantic":
            return self.semantic_chunking(text)
        elif method == "custom":
            return self.custom_chunking(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

chunker = DocumentChunker()