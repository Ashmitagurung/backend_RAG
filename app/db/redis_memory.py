import redis
import json
from typing import Dict, List, Optional
from app.config import settings

class RedisMemoryStore:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            print("Warning: Redis connection failed. Using in-memory fallback.")
            self.redis_client = None
            self._memory_store = {}
    
    def store_conversation(self, session_id: str, conversation: List[Dict]):
        """Store conversation history"""
        if self.redis_client:
            key = f"conversation:{session_id}"
            self.redis_client.setex(key, 3600, json.dumps(conversation))
        else:
            self._memory_store[f"conversation:{session_id}"] = conversation
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        if self.redis_client:
            key = f"conversation:{session_id}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else []
        else:
            return self._memory_store.get(f"conversation:{session_id}", [])
    
    def add_message(self, session_id: str, message: Dict):
        """Add single message to conversation"""
        conversation = self.get_conversation(session_id)
        conversation.append(message)
        self.store_conversation(session_id, conversation)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        if self.redis_client:
            key = f"conversation:{session_id}"
            self.redis_client.delete(key)
        else:
            self._memory_store.pop(f"conversation:{session_id}", None)

memory_store = RedisMemoryStore()