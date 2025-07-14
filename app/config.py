from pydantic_settings import BaseSettings
from typing import Optional, Set
import os

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./rag_app.db"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Vector DB (Pinecone)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "rag-index"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformer"
    EMBEDDING_DIMENSION: int = 384
    
    # OpenAI (Optional)
    OPENAI_API_KEY: Optional[str] = None
    
    # Email SMTP
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    
    # App settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt"}
    DEBUG: bool = False
    
    # Chunking defaults
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()