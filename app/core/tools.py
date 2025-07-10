# app/core/tools.py
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.vector_store import vector_store
from app.core.embedding import embedding_generator
from app.db.redis_memory import memory_store
from app.db.metadata_db import get_db
from app.db.models import BookingRequest
from app.utils.email_utils import send_booking_confirmation
from datetime import datetime
import json

class DocumentSearchInput(BaseModel):
    query: str = Field(description="The search query")
    top_k: int = Field(default=5, description="Number of results to return")
    method: str = Field(default="cosine", description="Similarity search method")

class BookingInput(BaseModel):
    full_name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(description="Time in HH:MM format")
    notes: str = Field(default="", description="Additional notes")

class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search through uploaded documents for relevant information"
    args_schema = DocumentSearchInput
    
    def _run(self, query: str, top_k: int = 5, method: str = "cosine") -> str:
        """Search documents for relevant information"""
        try:
            # Generate query embedding
            query_embeddings, _ = embedding_generator.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search vector store
            results, metrics = vector_store.similarity_search(
                query_embedding, 
                top_k=top_k,
                method=method
            )
            
            # Format results
            formatted_results = {
                "search_results": [],
                "metrics": metrics
            }
            
            for result in results:
                formatted_results["search_results"].append({
                    "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                    "score": result["score"],
                    "filename": result["metadata"].get("filename", "unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0)
                })
            
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    async def _arun(self, query: str, top_k: int = 5, method: str = "cosine") -> str:
        return self._run(query, top_k, method)

class BookingTool(BaseTool):
    name = "book_interview"
    description = "Book an interview appointment with provided details"
    args_schema = BookingInput
    
    def _run(self, full_name: str, email: str, date: str, time: str, notes: str = "") -> str:
        """Book an interview appointment"""
        try:
            # Validate date format
            booking_date = datetime.strptime(date, "%Y-%m-%d")
            
            # Validate time format
            datetime.strptime(time, "%H:%M")
            
            # Create booking in database
            db = next(get_db())
            try:
                booking = BookingRequest(
                    full_name=full_name,
                    email=email,
                    booking_date=booking_date,
                    booking_time=time,
                    notes=notes
                )
                
                db.add(booking)
                db.commit()
                db.refresh(booking)
                
                # Send confirmation email
                email_sent = send_booking_confirmation(email, full_name, date, time)
                
                result = {
                    "success": True,
                    "booking_id": booking.id,
                    "message": f"Interview booked successfully for {full_name} on {date} at {time}",
                    "email_sent": email_sent
                }
                
                return json.dumps(result, indent=2)
                
            finally:
                db.close()
                
        except ValueError as e:
            return f"Error: Invalid date or time format - {str(e)}"
        except Exception as e:
            return f"Error booking interview: {str(e)}"
    
    async def _arun(self, full_name: str, email: str, date: str, time: str, notes: str = "") -> str:
        return self._run(full_name, email, date, time, notes)