from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.metadata_db import get_db
from app.db.models import DocumentMetadata
from app.core.chunking import chunker
from app.core.embedding import embedding_generator
from app.core.vector_store import vector_store
from app.config import settings
import os
import uuid
from datetime import datetime
import PyPDF2
import io

router = APIRouter()

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file"""
    return file_content.decode('utf-8')

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = "recursive",
    embedding_model: str = "openai",
    db: Session = Depends(get_db)
):
    """Upload and process document"""
    
    # Validate file
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Read file content
    file_content = await file.read()
    
    # Extract text based on file type
    if file_extension == ".pdf":
        text = extract_text_from_pdf(file_content)
    elif file_extension == ".txt":
        text = extract_text_from_txt(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    # Chunk the document
    chunks = chunker.chunk_document(text, method=chunking_method)
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(chunks, model=embedding_model)
    
    # Prepare metadata for vector store
    chunk_metadata = [{
        "document_id": document_id,
        "filename": file.filename,
        "chunk_index": i,
        "chunking_method": chunking_method,
        "embedding_model": embedding_model
    } for i in range(len(chunks))]
    
    # Store in vector database
    vector_ids = vector_store.store_embeddings(embeddings, chunks, chunk_metadata)
    
    # Save metadata to relational database
    doc_metadata = DocumentMetadata(
        filename=file.filename,
        file_path=f"uploads/{document_id}_{file.filename}",
        chunking_method=chunking_method,
        embedding_model=embedding_model,
        total_chunks=len(chunks),
        file_size=file.size
    )
    
    db.add(doc_metadata)
    db.commit()
    db.refresh(doc_metadata)
    
    return {
        "document_id": document_id,
        "filename": file.filename,
        "total_chunks": len(chunks),
        "chunking_method": chunking_method,
        "embedding_model": embedding_model,
        "vector_ids": vector_ids[:5]  # Return first 5 for reference
    }

@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents"""
    documents = db.query(DocumentMetadata).all()
    return documents