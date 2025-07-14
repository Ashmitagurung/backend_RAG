from typing import List, Dict, Tuple, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
import uuid
import time
import logging
import os
from dotenv import load_dotenv  # ✅ load .env

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            api_key = os.getenv("PINECONE_API_KEY") or settings.PINECONE_API_KEY
            environment = os.getenv("PINECONE_ENVIRONMENT") or settings.PINECONE_ENVIRONMENT
            index_name = os.getenv("PINECONE_INDEX_NAME") or settings.PINECONE_INDEX_NAME
            embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", settings.EMBEDDING_DIMENSION))

            if not api_key:
                raise ValueError("❌ PINECONE_API_KEY is missing. Check your .env file.")
            
            self.pc = Pinecone(api_key=api_key)

            # List existing indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if index_name not in index_names:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                time.sleep(10)  # Wait for index to be created
            
            self.index = self.pc.Index(index_name)
            logger.info("✅ Pinecone initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            raise
    
    def test_connection(self):
        try:
            if self.index:
                stats = self.index.describe_index_stats()
                logger.info(f"Pinecone connection test successful. Index stats: {stats}")
                return True
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            raise
        return False

    # Other methods (store_embeddings, similarity_search, delete_by_document) remain unchanged
    # Keep them as in your current file — no change needed below this point

# Global instance
vector_store = VectorStore()
