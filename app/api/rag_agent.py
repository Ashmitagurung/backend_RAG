from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from app.core.tools import document_search_tool, booking_tool
from app.db.redis_memory import memory_store
from app.config import settings
import uuid

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
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=settings.OPENAI_API_KEY
        )
        
        self.tools = [document_search_tool, booking_tool]
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful RAG assistant that can:
            1. Search through uploaded documents to answer questions
            2. Book interview appointments when requested
            
            Always use the document_search tool to find relevant information before answering questions.
            For booking requests, use the book_interview tool with the provided details.
            
            Be conversational and helpful. If you can't find information in the documents, say so clearly.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, query: str, session_id: str) -> Dict:
        """Process user query with memory"""
        # Get conversation history
        conversation_history = memory_store.get_conversation(session_id)
        
        # Format chat history for LangChain
        chat_history = []
        for msg in conversation_history:
            if msg["role"] == "user":
                chat_history.append(("user", msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(("assistant", msg["content"]))
        
        # Run agent
        result = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # Update memory
        memory_store.add_message(session_id, {"role": "user", "content": query})
        memory_store.add_message(session_id, {"role": "assistant", "content": result["output"]})
        
        return {
            "response": result["output"],
            "session_id": session_id
        }

# Global agent instance
rag_agent = RAGAgent()

@router.post("/query", response_model=QueryResponse)
async def query_rag_agent(request: QueryRequest):
    """Query the RAG agent"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process query
        result = rag_agent.process_query(request.query, session_id)
        
        return QueryResponse(
            response=result["response"],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session"""
    memory_store.clear_conversation(session_id)
    return {"message": "Session cleared successfully"}