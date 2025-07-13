import redis
import json
from typing import Dict, List, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisMemoryStore:
    def __init__(self):
        self.redis_client = None
        self._memory_store = {}
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with fallback"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"⚠️ Redis connection failed, using in-memory fallback: {e}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"❌ Redis initialization error: {e}")
            self.redis_client = None
    
    def _get_conversation_key(self, session_id: str) -> str:
        """Get Redis key for conversation"""
        return f"conversation:{session_id}"
    
    def store_conversation(self, session_id: str, conversation: List[Dict]):
        """Store conversation history"""
        try:
            if self.redis_client:
                key = self._get_conversation_key(session_id)
                self.redis_client.setex(key, 3600, json.dumps(conversation))
            else:
                self._memory_store[self._get_conversation_key(session_id)] = conversation
        except Exception as e:
            logger.error(f"❌ Failed to store conversation: {e}")
            # Fallback to memory store
            self._memory_store[self._get_conversation_key(session_id)] = conversation
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        try:
            key = self._get_conversation_key(session_id)
            
            if self.redis_client:
                data = self.redis_client.get(key)
                return json.loads(data) if data else []
            else:
                return self._memory_store.get(key, [])
                
        except Exception as e:
            logger.error(f"❌ Failed to get conversation: {e}")
            return []
    
    def add_message(self, session_id: str, message: Dict):
        """Add single message to conversation"""
        try:
            conversation = self.get_conversation(session_id)
            conversation.append(message)
            # Keep only last 50 messages to prevent memory issues
            if len(conversation) > 50:
                conversation = conversation[-50:]
            self.store_conversation(session_id, conversation)
        except Exception as e:
            logger.error(f"❌ Failed to add message: {e}")
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        try:
            key = self._get_conversation_key(session_id)
            
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self._memory_store.pop(key, None)
                
        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {e}")
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys("conversation:*")
                return [key.replace("conversation:", "") for key in keys]
            else:
                keys = list(self._memory_store.keys())
                return [key.replace("conversation:", "") for key in keys if key.startswith("conversation:")]
        except Exception as e:
            logger.error(f"❌ Failed to get sessions: {e}")
            return []

# Global instance
memory_store = RedisMemoryStore()