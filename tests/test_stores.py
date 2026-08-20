"""
Session store and vector store integrity.

Both hold state that outlives a request, and both are where one user's data could reach
another. The checks here are mostly about things staying separate, staying in step, and
expiring when they should.
"""

import unittest
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd

import app.core.vectorstore as vs_mod
from app.core.config import get_settings
from app.core.session import SessionStore
from app.core.vectorstore import VectorStore

settings = get_settings()
DIM = 8


class FakeRedis:
    """
    In-memory stand-in for redis-py.

    Reads return bytes because the real client is built without decode_responses, and a
    fake that returns str would let a missing .decode() pass here and fail in production.
    """

    def __init__(self):
        self.store, self.lists, self.sets, self.hashes = {}, {}, {}, {}
        self.expiries = {}

    @staticmethod
    def _raw(value):
        return value.encode() if isinstance(value, str) else value

    async def ping(self):
        return True

    async def set(self, key, value):
        self.store[key] = self._raw(value)

    async def get(self, key):
        return self.store.get(key)

    async def expire(self, key, ttl):
        self.expiries[key] = ttl
        return True

    async def delete(self, *keys):
        for key in keys:
            for space in (self.store, self.lists, self.sets, self.hashes):
                space.pop(key, None)
        return len(keys)

    async def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(self._raw(v) for v in values)
        return len(self.lists[key])

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        self.lists[key] = items[start:] if end == -1 else items[start : end + 1]
        return True

    async def lrem(self, key, count, value):
        items = self.lists.get(key, [])
        raw = self._raw(value)
        if raw in items:
            items.remove(raw)
        return 1

    async def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(self._raw(v) for v in values)
        return len(values)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def srem(self, key, *values):
        self.sets.get(key, set()).difference_update(self._raw(v) for v in values)
        return len(values)

    async def hset(self, key, field=None, value=None, mapping=None):
        entry = self.hashes.setdefault(key, {})
        if mapping:
            entry.update({self._raw(k): self._raw(v) for k, v in mapping.items()})
        elif field is not None:
            entry[self._raw(field)] = self._raw(value)
        return 1

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(self._raw(field))

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hdel(self, key, *fields):
        entry = self.hashes.get(key, {})
        for field in fields:
            entry.pop(self._raw(field), None)
        return len(fields)

    def _all_keys(self):
        return [*self.store, *self.lists, *self.sets, *self.hashes]

    async def keys(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self._all_keys() if k.startswith(prefix)]

    async def exists(self, key):
        return int(key in self._all_keys())

    async def scan_iter(self, match=None):
        prefix = (match or "").rstrip("*")
        for key in self._all_keys():
            if key.startswith(prefix):
                yield key


def session_store():
    store = SessionStore.__new__(SessionStore)
    store.redis_client = FakeRedis()
    store._initialized = True
    return store


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), DIM), dtype="float32")


def vector_store():
    store = VectorStore()
    store.dim = DIM
    # Never write to the configured index directory. Without this the suite persisted an
    # 8-dimensional index over the real one; the next start found the dimensions
    # disagreed, discarded it, and the knowledge base came up empty.
    store.persist = False
    store._initialized = True
    store.index = faiss.IndexFlatIP(DIM)
    store.embedder = FakeEmbedder()
    store.documents = []
    return store


def unit_vectors(n):
    vectors = np.random.rand(n, DIM).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors


class SessionsStaySeparate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = session_store()
        await self.store.save_dataframe("alice", "sales", pd.DataFrame({"a": [1]}))
        await self.store.save_dataframe("bob", "sales", pd.DataFrame({"a": [99]}))

    async def test_the_same_table_name_holds_different_data_per_session(self):
        alice = await self.store.get_dataframe("alice", "sales")
        bob = await self.store.get_dataframe("bob", "sales")
        self.assertEqual(alice["a"].iloc[0], 1)
        self.assertEqual(bob["a"].iloc[0], 99)

    async def test_a_stranger_gets_nothing_rather_than_someone_elses_table(self):
        self.assertIsNone(await self.store.get_dataframe("carol", "sales"))

    async def test_the_table_list_is_per_session(self):
        self.assertEqual(await self.store.list_dataframes("alice"), ["sales"])
        self.assertEqual(await self.store.list_dataframes("carol"), [])

    async def test_history_is_invisible_across_sessions(self):
        await self.store.add_chat_turn("alice", "q", "a")
        self.assertEqual(await self.store.get_chat_history("bob"), [])


class HistoryIsBounded(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = session_store()
        self.total = settings.MAX_CHAT_HISTORY + 8
        for i in range(self.total):
            await self.store.add_chat_turn("alice", f"q{i}", f"a{i}")

    async def test_history_is_trimmed_to_the_configured_limit(self):
        history = await self.store.get_chat_history("alice")
        self.assertLessEqual(len(history), settings.MAX_CHAT_HISTORY * 2)

    async def test_trimming_keeps_the_newest_turn(self):
        """Dropping from the wrong end would silently lose the current conversation."""
        history = await self.store.get_chat_history("alice")
        self.assertTrue(history[-1]["content"].endswith(str(self.total - 1)))

    async def test_every_message_has_a_role(self):
        history = await self.store.get_chat_history("alice")
        self.assertLessEqual({m["role"] for m in history}, {"user", "assistant"})

    async def test_clearing_history_leaves_the_tables_alone(self):
        await self.store.save_dataframe("alice", "t", pd.DataFrame({"a": [1]}))
        await self.store.clear_chat_history("alice")
        self.assertEqual(await self.store.get_chat_history("alice"), [])
        self.assertIsNotNone(await self.store.get_dataframe("alice", "t"))


class ExpiryPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_a_user_session_is_given_a_ttl(self):
        store = session_store()
        await store.add_chat_turn("user1", "hi", "hello")
        self.assertTrue(any("user1" in key for key in store.redis_client.expiries))

    async def test_the_shared_knowledge_base_never_expires(self):
        """It is a corpus, not a conversation; expiring it would empty the KB silently."""
        store = session_store()
        await store.add_chat_turn(settings.KB_SESSION_ID, "hi", "hello")
        self.assertEqual(store.redis_client.expiries, {})


class DeletionIsThorough(unittest.IsolatedAsyncioTestCase):
    class RecordingStore:
        def __init__(self):
            self.by_source, self.by_session = [], []

        async def remove_by_source(self, source, session_id=None):
            self.by_source.append((source, session_id))
            return 1

        async def remove_by_session(self, session_id):
            self.by_session.append(session_id)
            return 1

    async def test_deleting_a_document_takes_its_tables_and_vectors(self):
        store = session_store()
        await store.save_dataframe("alice", "sales", pd.DataFrame({"a": [1]}))
        await store.register_document(
            "alice", "sales.csv", "doc-1",
            {"tables": ["sales"], "text_chunks": 1, "dataframes": 1},
        )

        vectors = self.RecordingStore()
        await store.delete_document("alice", "doc-1", vector_store=vectors)

        self.assertIsNone(await store.get_dataframe("alice", "sales"))
        self.assertEqual(vectors.by_source, [("sales.csv", "alice")],
                         "the removal must name the session, not just the file")
        self.assertEqual(await store.get_session_documents("alice"), [])

    async def test_clearing_one_session_does_not_touch_another(self):
        store = session_store()
        await store.save_dataframe("alice", "t", pd.DataFrame({"a": [1]}))
        await store.add_chat_turn("alice", "q", "a")
        await store.save_dataframe("bob", "t", pd.DataFrame({"a": [2]}))

        vectors = self.RecordingStore()
        await store.clear_session("alice", vector_store=vectors)

        self.assertIsNone(await store.get_dataframe("alice", "t"))
        self.assertEqual(await store.get_chat_history("alice"), [])
        self.assertEqual(vectors.by_session, ["alice"])
        self.assertIsNotNone(await store.get_dataframe("bob", "t"))


class RegistryRecordsTheFacts(unittest.IsolatedAsyncioTestCase):
    async def test_a_document_is_registered_and_scoped(self):
        store = session_store()
        await store.register_document(
            "alice", "sales.csv", "doc-1",
            {"tables": ["sales"], "text_chunks": 3, "dataframes": 1},
        )
        documents = await store.get_session_documents("alice")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["doc_id"], "doc-1")
        self.assertEqual(documents[0]["filename"], "sales.csv")
        self.assertEqual(await store.get_session_documents("bob"), [])

    async def test_stats_count_what_is_stored(self):
        store = session_store()
        await store.register_document("alice", "s.csv", "doc-1", {"tables": ["s"]})
        self.assertEqual((await store.get_session_stats("alice"))["documents"], 1)


class VectorSearchRespectsScope(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = vector_store()
        self.store.documents = [
            {"content": f"doc {i}", "source": "s.pdf", "session_id": sid}
            for i, sid in enumerate(["alice", "alice", "bob", "kb_default"])
        ]
        self.store.index.add(unit_vectors(4))

    async def test_only_permitted_positions_come_back(self):
        hits = await self.store.search("anything", 10, allowed={0, 1})
        self.assertTrue(hits)
        self.assertTrue(all(index in (0, 1) for index, _ in hits))

    async def test_an_unrestricted_search_sees_everything(self):
        self.assertEqual(len(await self.store.search("anything", 10, allowed=None)), 4)

    async def test_an_empty_permission_set_returns_nothing(self):
        self.assertEqual(await self.store.search("anything", 10, allowed=set()), [])


class IndexAndDocumentsStayInStep(unittest.IsolatedAsyncioTestCase):
    """
    A divergence here misattributes every later hit: position 3 in the index would name
    whatever document happens to sit at position 3 in the list.
    """

    async def asyncSetUp(self):
        self.store = vector_store()
        self.persist = patch.object(vs_mod.settings, "PERSIST_INDEX", False)
        self.persist.start()
        self.addCleanup(self.persist.stop)
        await self.store.add_documents([
            {"content": f"d{i}", "source": f"s{i % 2}.pdf", "session_id": "alice"}
            for i in range(6)
        ])

    def assertInStep(self):
        self.assertEqual(self.store.index.ntotal, len(self.store.documents))

    async def test_they_agree_after_a_batch_add(self):
        self.assertInStep()

    async def test_they_agree_after_a_removal_by_source(self):
        await self.store.remove_by_source("s0.pdf")
        self.assertInStep()
        self.assertEqual({d["source"] for d in self.store.documents}, {"s1.pdf"})

    async def test_they_agree_after_a_removal_by_session(self):
        await self.store.add_documents(
            [{"content": "x", "source": "b.pdf", "session_id": "bob"}]
        )
        await self.store.remove_by_session("bob")
        self.assertInStep()
        self.assertNotIn("bob", {d["session_id"] for d in self.store.documents})

    async def test_clearing_empties_both(self):
        await self.store.clear()
        self.assertEqual(self.store.documents, [])
        self.assertTrue(self.store.index is None or self.store.index.ntotal == 0)

    async def test_every_mutation_bumps_the_version(self):
        """The BM25 cache keys on this; a missed bump serves a stale corpus."""
        before = self.store.version
        await self.store.add_documents(
            [{"content": "new", "source": "n.pdf", "session_id": "alice"}]
        )
        after_add = self.store.version
        self.assertGreater(after_add, before)

        await self.store.remove_by_source("n.pdf")
        self.assertGreater(self.store.version, after_add)


class PruningKeepsTheCorpus(unittest.IsolatedAsyncioTestCase):
    async def test_expired_sessions_go_and_the_knowledge_base_stays(self):
        store = vector_store()
        store.documents = [
            {"content": f"d{i}", "source": "s.pdf", "session_id": sid}
            for i, sid in enumerate(["alive", "dead", "alive", "dead", settings.KB_SESSION_ID])
        ]
        store.index.add(unit_vectors(5))

        with patch.object(vs_mod.settings, "PERSIST_INDEX", False):
            removed = await store.prune_sessions(lambda sid: sid != "dead")

        self.assertEqual(removed, 2)
        self.assertEqual(store.index.ntotal, 3)
        self.assertEqual(
            {d["session_id"] for d in store.documents},
            {"alive", settings.KB_SESSION_ID},
        )

    async def test_stats_report_documents_and_vectors_consistently(self):
        store = vector_store()
        store.documents = [{"content": "d", "source": "s.pdf", "session_id": "alice"}]
        store.index.add(unit_vectors(1))
        stats = store.get_stats()
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["indexed_vectors"], 1)
        self.assertEqual(stats["dimension"], DIM)


if __name__ == "__main__":
    unittest.main()


class DeletingOneSessionsFileLeavesAnothers(unittest.IsolatedAsyncioTestCase):
    """
    `remove_by_source` matched on filename alone. Two people uploading `report.pdf` own
    different documents, so deleting either took both — and re-uploading a corrected file,
    which supersedes by filename, did the same to a stranger's copy.
    """

    async def test_only_the_owning_sessions_chunks_go(self):
        store = vector_store()
        store.documents = [
            {"content": "alice's report", "source": "report.pdf", "session_id": "alice"},
            {"content": "bob's report", "source": "report.pdf", "session_id": "bob"},
            {"content": "alice's other", "source": "notes.pdf", "session_id": "alice"},
        ]
        store.index.add(unit_vectors(3))

        with patch.object(vs_mod.settings, "PERSIST_INDEX", False):
            removed = await store.remove_by_source("report.pdf", session_id="alice")

        self.assertEqual(removed, 1)
        self.assertEqual(store.index.ntotal, 2)
        remaining = {(d["session_id"], d["source"]) for d in store.documents}
        self.assertIn(("bob", "report.pdf"), remaining)

    async def test_an_unscoped_removal_still_takes_every_copy(self):
        """The knowledge-base refresh relies on this: there the filename is the identity."""
        store = vector_store()
        store.documents = [
            {"content": "a", "source": "kb.pdf", "session_id": "kb_default"},
            {"content": "b", "source": "kb.pdf", "session_id": "kb_default"},
            {"content": "c", "source": "other.pdf", "session_id": "kb_default"},
        ]
        store.index.add(unit_vectors(3))

        with patch.object(vs_mod.settings, "PERSIST_INDEX", False):
            removed = await store.remove_by_source("kb.pdf")

        self.assertEqual(removed, 2)
        self.assertEqual(store.index.ntotal, 1)
