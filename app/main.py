# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, rag_agent, booking
from app.db.metadata_db import create_tables
from app.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="RAG Backend API",
    description="A RAG-based backend with document processing and booking system",
    version="1.0.0"
)

# Run this at startup to create database tables
create_tables()

# Enable CORS (configure properly for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with allowed frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(rag_agent.router, prefix="/api/v1/rag", tags=["RAG Agent"])
app.include_router(booking.router, prefix="/api/v1/booking", tags=["Booking"])

# Root route
@app.get("/")
async def root():
    return {"message": "RAG Backend API is running"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
