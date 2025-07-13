# app/api/rag_agent.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.core.tools import document_search_tool, booking_tool
from app.main import memory_store
import uuid
from transformers import pipeline

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict] = []

class RAGAgent:
    def __init__(self):
        # Load HuggingFace model (e.g., Flan-T5)
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base")

        # Available tools (just for documentation logic here)
        self.tools = {
            "document_search": document_search_tool,
            "book_interview": booking_tool,
        }

    def process_query(self, query: str, session_id: str) -> Dict:
        """Simple RAG response without LangChain agent"""
        # Add conversation history
        memory_store.add_message(session_id, {"role": "user", "content": query})

        # Very simple agent logic (call document search tool before LLM)
        search_result = self.tools["document_search"].run({"query": query})

        prompt = f"""
You are a helpful assistant. First read the following context from document search:
{search_result.get("result", "No relevant document found")}

Then answer the user question: {query}
"""

        llm_response = self.llm(prompt, max_length=256, do_sample=False)[0]["generated_text"]

        memory_store.add_message(session_id, {"role": "assistant", "content": llm_response})

        return {
            "response": llm_response,
            "session_id": session_id,
            "sources": [{"source": "vector_store"}]
        }

rag_agent_instance = RAGAgent()

@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        result = rag_agent_instance.process_query(request.query, session_id)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    memory_store.clear_conversation(session_id)
    return {"message": "Session cleared"}
