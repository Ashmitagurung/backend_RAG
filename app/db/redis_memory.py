import redis
import json
from typing import Dict, List, Optional
from app.config import settings

class RedisMemoryStore:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    def store_conversation(self, session_id: str, conversation: List[Dict]):
        """Store conversation history"""
        key = f"conversation:{session_id}"
        self.redis_client.setex(key, 3600, json.dumps(conversation))  # 1 hour TTL
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        key = f"conversation:{session_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else []
    
    def add_message(self, session_id: str, message: Dict):
        """Add single message to conversation"""
        conversation = self.get_conversation(session_id)
        conversation.append(message)
        self.store_conversation(session_id, conversation)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        key = f"conversation:{session_id}"
        self.redis_client.delete(key)

memory_store = RedisMemoryStore()