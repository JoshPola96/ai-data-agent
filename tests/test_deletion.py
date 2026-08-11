"""
Deleting a document must remove everything it produced.

Both regressions here were found by driving the real UI: the registry entry and the
vectors were removed while the extracted tables were left behind, and the uploader
re-ingested the file on the very next rerun.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.core.session import SessionStore


class FakeRedis:
    """Enough of Redis to exercise the deletion path."""

    def __init__(self):
        self.hashes = {}
        self.values = {}
        self.sets = {}

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value.encode()

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hdel(self, key, field):
        self.hashes.get(key, {}).pop(field, None)

    async def hgetall(self, key):
        return self.hashes.get(key, {})

    async def set(self, key, value):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, *keys):
        for k in keys:
            self.values.pop(k, None)
            self.hashes.pop(k, None)
            self.sets.pop(k, None)

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def srem(self, key, member):
        self.sets.get(key, set()).discard(member)

    async def smembers(self, key):
        return {m.encode() for m in self.sets.get(key, set())}

    async def expire(self, key, ttl):
        return True

    async def exists(self, key):
        return int(key in self.values or key in self.hashes or key in self.sets)

    async def rpush(self, key, value):
        self.values.setdefault(key, []).append(value)

    async def ltrim(self, key, start, end):
        return True

    async def lrange(self, key, start, end):
        return self.values.get(key, [])

    async def ping(self):
        return True


class DeleteDocumentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = SessionStore()
        self.store.redis_client = FakeRedis()
        self.session = "s1"

        # A document that produced two tables, as an Excel workbook would
        await self.store.register_document(
            self.session,
            "payslips.xlsx",
            "doc1",
            {"text_chunks": 3, "dataframes": 2, "tables": ["payslips_Jan", "payslips_Feb"]},
        )
        for name in ("payslips_Jan", "payslips_Feb"):
            await self.store.redis_client.sadd(self.store._df_list_key(self.session), name)
            await self.store.redis_client.set(self.store._df_key(self.session, name), b"x")

    async def test_tables_are_removed_with_the_document(self):
        vs = MagicMock()
        vs.remove_by_source = AsyncMock(return_value=3)

        await self.store.delete_document(self.session, "doc1", vector_store=vs)

        self.assertEqual(await self.store.list_dataframes(self.session), [])

    async def test_the_registry_entry_is_removed(self):
        vs = MagicMock()
        vs.remove_by_source = AsyncMock(return_value=3)

        await self.store.delete_document(self.session, "doc1", vector_store=vs)

        self.assertEqual(await self.store.get_session_documents(self.session), [])

    async def test_vectors_are_removed_by_source(self):
        vs = MagicMock()
        vs.remove_by_source = AsyncMock(return_value=3)

        await self.store.delete_document(self.session, "doc1", vector_store=vs)

        vs.remove_by_source.assert_awaited_once_with("payslips.xlsx")

    async def test_deleting_an_unknown_document_is_reported(self):
        self.assertFalse(await self.store.delete_document(self.session, "ghost"))

    async def test_a_document_with_no_tables_deletes_cleanly(self):
        await self.store.register_document(
            self.session, "notes.txt", "doc2", {"text_chunks": 1, "dataframes": 0}
        )
        vs = MagicMock()
        vs.remove_by_source = AsyncMock(return_value=1)

        self.assertTrue(await self.store.delete_document(self.session, "doc2", vector_store=vs))

    async def test_other_documents_tables_are_untouched(self):
        await self.store.register_document(
            self.session, "other.csv", "doc2", {"text_chunks": 1, "dataframes": 1, "tables": ["other"]}
        )
        await self.store.redis_client.sadd(self.store._df_list_key(self.session), "other")
        await self.store.redis_client.set(self.store._df_key(self.session, "other"), b"y")

        vs = MagicMock()
        vs.remove_by_source = AsyncMock(return_value=3)
        await self.store.delete_document(self.session, "doc1", vector_store=vs)

        self.assertEqual(await self.store.list_dataframes(self.session), ["other"])


class KnowledgeBaseTtlTest(unittest.IsolatedAsyncioTestCase):
    """The shared corpus is not a user session and must not expire."""

    async def asyncSetUp(self):
        self.store = SessionStore()
        self.store.redis_client = FakeRedis()
        self.expiries = []
        original = self.store.redis_client.expire

        async def record(key, ttl):
            self.expiries.append(key)
            return await original(key, ttl)

        self.store.redis_client.expire = record

    async def test_a_user_session_key_is_given_a_ttl(self):
        await self.store.add_chat_turn("user1", "hi", "hello")
        self.assertIn("chat:user1", self.expiries)

    async def test_the_knowledge_base_key_is_not(self):
        from app.core.config import get_settings

        await self.store.add_chat_turn(get_settings().KB_SESSION_ID, "hi", "hello")
        self.assertEqual(self.expiries, [])


if __name__ == "__main__":
    unittest.main()
