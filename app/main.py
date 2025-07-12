# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from app.api import upload, rag_agent, booking
from app.db.metadata_db import create_tables
from app.config import settings

from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into os.environ


# Create FastAPI app instance
app = FastAPI(
    title="RAG Backend API",
    description="A RAG-based backend with document processing and booking system",
    version="1.0.0"
)

# Enable CORS (configure properly for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Event handler for startup tasks like DB initialization
@app.on_event("startup")
async def startup_event():
    try:
        create_tables()
        print("✅ Database tables created successfully.")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")

# Include API routers
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(rag_agent.router, prefix="/api/v1/rag", tags=["RAG Agent"])
app.include_router(booking.router, prefix="/api/v1/booking", tags=["Booking"])

# Root route
@app.get("/")
async def root():
    return {"message": "RAG Backend API is running 🚀"}

# Health check route
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
