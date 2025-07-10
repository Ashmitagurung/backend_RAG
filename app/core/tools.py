from typing import List, Dict, Any
from langchain.tools import BaseTool
from app.core.vector_store import vector_store
from app.core.embedding import embedding_generator
from app.db.redis_memory import memory_store
from app.db.metadata_db import get_db
from app.db.models import BookingRequest
from app.utils.email_utils import send_booking_confirmation
from datetime import datetime
import json

class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search through uploaded documents for relevant information"
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Search documents for relevant information"""
        # Generate query embedding
        query_embedding = embedding_generator.generate_embeddings([query])[0]
        
        # Search vector store
        results = vector_store.similarity_search(query_embedding, top_k=top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"],
                "score": result["score"],
                "filename": result["metadata"].get("filename", "unknown"),
                "chunk_index": result["metadata"].get("chunk_index", 0)
            })
        
        return json.dumps(formatted_results, indent=2)
    
    async def _arun(self, query: str, top_k: int = 5) -> str:
        return self._run(query, top_k)

class BookingTool(BaseTool):
    name = "book_interview"
    description = "Book an interview appointment with provided details"
    
    def _run(self, full_name: str, email: str, date: str, time: str, notes: str = "") -> str:
        """Book an interview appointment"""
        try:
            # Parse date
            booking_date = datetime.strptime(date, "%Y-%m-%d")
            
            # Create booking in database
            db = next(get_db())
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
            send_booking_confirmation(email, full_name, date, time)
            
            return f"Interview booked successfully for {full_name} on {date} at {time}. Confirmation email sent."
            
        except Exception as e:
            return f"Error booking interview: {str(e)}"
    
    async def _arun(self, full_name: str, email: str, date: str, time: str, notes: str = "") -> str:
        return self._run(full_name, email, date, time, notes)

# Tool instances
document_search_tool = DocumentSearchTool()
booking_tool = BookingTool()