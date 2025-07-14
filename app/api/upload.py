from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.metadata_db import get_db
from app.db.models import DocumentMetadata
from app.core.chunking import chunker
from app.core.embedding import embedding_generator
from app.config import settings

import os
import uuid
from datetime import datetime
import PyPDF2
import io

# ✅ Use shared vector_store instance to avoid re-initializing Pinecone every request
from app.core.vector_store import vector_store

router = APIRouter()

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    return ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_txt(file_content: bytes) -> str:
    return file_content.decode('utf-8')

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = "recursive",
    embedding_model: str = settings.EMBEDDING_MODEL,
    db: Session = Depends(get_db)
):
    # ✅ 1. Validate file type
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    extension = os.path.splitext(file.filename)[1].lower()

    if extension == ".pdf":
        text = extract_text_from_pdf(content)
    elif extension == ".txt":
        text = extract_text_from_txt(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported extension")

    # ✅ 2. Generate document ID
    document_id = str(uuid.uuid4())

    # ✅ 3. Chunk and embed
    chunks, _ = chunker.chunk_document(text, method=chunking_method)
    embeddings, _ = embedding_generator.generate_embeddings(chunks, model=embedding_model)

    # ✅ 4. Prepare metadata
    metadata = [
        {
            "document_id": document_id,
            "filename": file.filename,
            "chunk_index": i,
            "chunking_method": chunking_method,
            "embedding_model": embedding_model
        } for i in range(len(chunks))
    ]

    # ✅ 5. Store vectors in Pinecone
    vector_ids = vector_store.store_embeddings(embeddings, chunks, metadata)

    # ✅ 6. Save file locally
    file_path = f"uploads/{document_id}_{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    # ✅ 7. Store metadata in DB
    doc_record = DocumentMetadata(
        document_id=document_id,
        filename=file.filename,
        file_path=file_path,
        chunking_method=chunking_method,
        embedding_model=embedding_model,
        total_chunks=len(chunks),
        file_size=len(content),
        processed=True
    )

    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "total_chunks": len(chunks),
        "vector_ids": vector_ids[:5]  # Show preview
    }
