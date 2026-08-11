# app/core/session.py

"""
Session Management
Redis-backed session store for chat history, dataframe storage, and document tracking
"""

import json
import logging
import pickle
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pandas as pd
import redis.asyncio as redis
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Enforce 1-hour session timeout (3600 seconds)
SESSION_TTL = 3600


class SessionStore:
    """Redis-backed session management with multi-document support"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize Redis connection"""
        if self.redis_client is None:
            try:
                self.redis_client = await redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=False,  # Binary mode for pickle
                )
                # Test connection
                await self.redis_client.ping()
                logger.info(f"✅ Connected to Redis: {settings.REDIS_URL}")
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                raise

    async def _touch(self, key: str, session_id: str):
        """
        Refresh a session key's lifetime.

        The shared knowledge base is not a user session — it is the corpus everyone
        retrieves from — so its keys never expire. Letting them lapse left vectors
        on disk with no document registry describing them.
        """
        if session_id != settings.KB_SESSION_ID:
            await self.redis_client.expire(key, SESSION_TTL)

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔌 Redis connection closed")

    # =========================
    # Chat History
    # =========================

    def _chat_key(self, session_id: str) -> str:
        """Generate Redis key for chat history"""
        return f"chat:{session_id}"

    async def add_chat_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        """Add a conversation turn to history"""
        await self.initialize()

        key = self._chat_key(session_id)
        turn = json.dumps(
            {
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.debug(f"💬 Adding chat turn to session {session_id[:8]}")

        # Add to list
        await self.redis_client.rpush(key, turn)

        # Trim to max history
        await self.redis_client.ltrim(key, -settings.MAX_CHAT_HISTORY, -1)

        # Set expiration (1 hour reset)
        await self._touch(key, session_id)

    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get chat history as list of message dicts.
        """
        await self.initialize()

        key = self._chat_key(session_id)

        # Reset TTL on access (Keep-Alive)
        await self._touch(key, session_id)

        raw_history = await self.redis_client.lrange(key, 0, -1)

        if not raw_history:
            return []

        # Convert to message format
        messages = []
        for turn_bytes in raw_history:
            turn = json.loads(turn_bytes.decode("utf-8"))
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        return messages

    async def clear_chat_history(self, session_id: str):
        """Clear chat history for a session"""
        await self.initialize()
        key = self._chat_key(session_id)
        await self.redis_client.delete(key)
        logger.info(f"🗑️ Cleared chat history for session {session_id[:8]}")

    # =========================
    # DataFrame Storage
    # =========================

    def _df_key(self, session_id: str, table_name: str) -> str:
        """Generate Redis key for dataframe"""
        return f"df:{session_id}:{table_name}"

    def _df_list_key(self, session_id: str) -> str:
        """Generate Redis key for dataframe list"""
        return f"df_list:{session_id}"

    async def save_dataframe(self, session_id: str, table_name: str, df: pd.DataFrame):
        """Save dataframe to Redis using pickle"""
        await self.initialize()

        # Serialize dataframe
        df_bytes = pickle.dumps(df)

        # Store dataframe with 1h TTL
        key = self._df_key(session_id, table_name)
        await self.redis_client.set(key, df_bytes)
        await self._touch(key, session_id)

        # Track in list with 1h TTL
        list_key = self._df_list_key(session_id)
        await self.redis_client.sadd(list_key, table_name)
        await self._touch(list_key, session_id)

        logger.info(
            f"✅ Saved dataframe '{table_name}' ({df.shape[0]}x{df.shape[1]}) for session {session_id[:8]}"
        )

    async def get_dataframe(
        self, session_id: str, table_name: str
    ) -> Optional[pd.DataFrame]:
        """Retrieve dataframe from Redis"""
        await self.initialize()

        key = self._df_key(session_id, table_name)

        # Reset TTL on access (Keep-Alive)
        await self._touch(key, session_id)

        df_bytes = await self.redis_client.get(key)

        if not df_bytes:
            return None

        return pickle.loads(df_bytes)

    async def get_all_dataframes(self, session_id: str) -> Dict[str, pd.DataFrame]:
        """Get all dataframes for a session"""
        await self.initialize()

        list_key = self._df_list_key(session_id)

        # Reset TTL on access
        await self._touch(list_key, session_id)

        table_names = await self.redis_client.smembers(list_key)

        if not table_names:
            return {}

        # Retrieve all dataframes
        dataframes = {}
        for name_bytes in table_names:
            name = name_bytes.decode("utf-8")
            df = await self.get_dataframe(session_id, name)
            if df is not None:
                dataframes[name] = df

        return dataframes

    async def list_dataframes(self, session_id: str) -> List[str]:
        """List all dataframe names for a session"""
        await self.initialize()

        list_key = self._df_list_key(session_id)
        await self._touch(list_key, session_id)

        names = await self.redis_client.smembers(list_key)
        return [n.decode("utf-8") for n in names] if names else []

    async def delete_dataframe(self, session_id: str, table_name: str):
        """Delete a specific dataframe"""
        await self.initialize()

        key = self._df_key(session_id, table_name)
        list_key = self._df_list_key(session_id)

        await self.redis_client.delete(key)
        await self.redis_client.srem(list_key, table_name)

    # =========================
    # Document Tracking (Multi-document support)
    # =========================

    def _doc_index_key(self, session_id: str) -> str:
        """Generate Redis key for document index"""
        return f"doc_index:{session_id}"

    async def register_document(
        self, session_id: str, filename: str, doc_id: str, metadata: Dict
    ):
        """Register a document in the session's document index"""
        await self.initialize()

        key = self._doc_index_key(session_id)

        doc_info = {
            "filename": filename,
            "doc_id": doc_id,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }

        # Store as hash and reset TTL to 1 hour
        await self.redis_client.hset(key, doc_id, json.dumps(doc_info))
        await self._touch(key, session_id)

        logger.info(f"📝 Registered document '{filename}' for session {session_id[:8]}")

    async def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents registered in a session"""
        await self.initialize()

        key = self._doc_index_key(session_id)

        # Reset TTL on access (Keep-Alive)
        await self._touch(key, session_id)

        doc_data = await self.redis_client.hgetall(key)

        if not doc_data:
            return []

        documents = []
        for doc_id, info_bytes in doc_data.items():
            info = json.loads(info_bytes.decode("utf-8"))
            documents.append(info)

        return documents

    async def delete_document(self, session_id: str, doc_id: str, vector_store=None):
        """
        Delete a specific document and its associated data/vectors
        """
        await self.initialize()
        logger.info(f"🗑️ Deleting document {doc_id} from session {session_id[:8]}")

        # 1. Get doc metadata to find filename/tables
        key = self._doc_index_key(session_id)
        doc_json = await self.redis_client.hget(key, doc_id)

        if not doc_json:
            logger.warning(f"⚠️ Document {doc_id} not found in index")
            return False

        doc_info = json.loads(doc_json)
        filename = doc_info.get("filename")

        # 2. Remove from Document Index
        await self.redis_client.hdel(key, doc_id)

        # 3. Remove the tables it produced. Without this a deleted document's data
        # stays queryable and chartable, which is wrong for a spreadsheet and worse
        # for a payslip.
        for table in doc_info.get("tables", []):
            await self.delete_dataframe(session_id, table)
            logger.info(f"🗑️ Removed table '{table}'")

        # 4. Remove from Vector Store (Physical)
        if vector_store and filename:
            try:
                # We assume Source field in vector store = filename
                count = await vector_store.remove_by_source(filename)
                logger.info(f"✅ Removed {count} vectors for {filename}")
            except Exception as e:
                logger.error(f"❌ Vector deletion failed: {e}")

        return True

    async def clear_session(self, session_id: str, vector_store=None):
        """
        Clear all session data including Redis state and Vector embeddings
        """
        await self.initialize()
        logger.info(f"🗑️ Deep cleaning session {session_id[:8]}...")

        # 1. Clear Chat & Dataframes (Redis)
        await self.clear_chat_history(session_id)

        table_names = await self.list_dataframes(session_id)
        for name in table_names:
            await self.delete_dataframe(session_id, name)

        # 2. Clear Document Index (Redis)
        await self.redis_client.delete(self._doc_index_key(session_id))
        await self.redis_client.delete(self._df_list_key(session_id))

        # 3. Clear Vector Store (Physical DB)
        if vector_store:
            try:
                await vector_store.remove_by_session(session_id)
                logger.info(f"✅ Removed vectors for session {session_id[:8]}")
            except Exception as e:
                logger.error(f"⚠️ Failed to clear vectors: {e}")

        logger.info(f"✨ Session {session_id[:8]} is now a clean slate.")

    async def session_exists(self, session_id: str) -> bool:
        """True while the session still holds state in Redis, i.e. has not expired."""
        await self.initialize()
        keys = (
            self._doc_index_key(session_id),
            self._chat_key(session_id),
            self._df_list_key(session_id),
        )
        return any([await self.redis_client.exists(k) for k in keys])

    async def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        await self.initialize()

        history = await self.get_chat_history(session_id)
        dataframes = await self.list_dataframes(session_id)
        documents = await self.get_session_documents(session_id)

        return {
            "session_id": session_id,
            "chat_turns": len(history) // 2,
            "dataframes": len(dataframes),
            "documents": len(documents),
            "document_list": [d["filename"] for d in documents],
        }
