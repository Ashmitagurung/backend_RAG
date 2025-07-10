from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, rag_agent, booking
from app.db.metadata_db import create_tables
from app.config import settings
import uvicorn

# Create tables on startup
create_tables()

app = FastAPI(
    title="RAG Backend API",
    description="A RAG-based backend with document processing and booking system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(rag_agent.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(booking.router, prefix="/api/v1/booking", tags=["booking"])

@app.get("/")
async def root():
    return {"message": "RAG Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
