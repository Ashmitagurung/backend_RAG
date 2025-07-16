from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

# Local imports
from app.api import upload, rag_agent, booking
from app.db.metadata_db import create_tables
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Create database tables
        create_tables()
        logger.info("Database tables created successfully")
        
        # Test connections
        from app.db.redis_memory import memory_store
        from app.core.vector_store import vector_store
        
        # Test Redis connection
        try:
            memory_store.redis_client.ping() if memory_store.redis_client else None
            logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f" Redis connection failed, using in-memory fallback: {e}")
        
        # Test Pinecone connection
        try:
            vector_store.test_connection()
            logger.info(" Pinecone connection successful")
        except Exception as e:
            logger.error(f" Pinecone connection failed: {e}")
            
        logger.info(" Application startup complete")
        
    except Exception as e:
        logger.error(f" Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info(" Application shutdown")

# Create FastAPI app instance
app = FastAPI(
    title="RAG Backend API",
    description="A production-ready RAG backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(rag_agent.router, prefix="/api/v1/rag", tags=["RAG Agent"])
app.include_router(booking.router, prefix="/api/v1/booking", tags=["Booking"])

# Root route
@app.get("/")
async def root():
    return {
        "message": "RAG Backend API is running ",
        "version": "1.0.0",
        "status": "healthy"
    }

# Health check route
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "database": "connected",
        "redis": "unknown",
        "vector_db": "unknown"
    }
    
    # Check Redis
    try:
        from app.db.redis_memory import memory_store
        if memory_store.redis_client:
            memory_store.redis_client.ping()
            health_status["redis"] = "connected"
        else:
            health_status["redis"] = "fallback"
    except Exception:
        health_status["redis"] = "disconnected"
    
    # Check Pinecone
    try:
        from app.core.vector_store import vector_store
        vector_store.test_connection()
        health_status["vector_db"] = "connected"
    except Exception:
        health_status["vector_db"] = "disconnected"
    
    return health_status

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error occurred"
    )