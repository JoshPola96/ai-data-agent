"""Index persistence: surviving restart, rejecting torn writes, pruning dead sessions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

import app.core.vectorstore as vs_mod
from app.core.vectorstore import VectorStore

DIM = 8


def fake_store(index_dir):
    """A VectorStore with the embedding model stubbed out; only I/O is under test."""
    store = VectorStore()
    store.dim = DIM
    store._initialized = True
    store.index = faiss.IndexFlatIP(DIM)
    store.embedder = object()
    return store


def vectors(n):
    v = np.random.rand(n, DIM).astype("float32")
    faiss.normalize_L2(v)
    return v


class PersistenceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.multiple(
            vs_mod.settings, PERSIST_INDEX=True, INDEX_DIR=self.tmp.name, EMBEDDING_DIM=DIM
        )
        p.start()
        self.addCleanup(p.stop)

    def populated(self, n=4, session="alice"):
        store = fake_store(self.tmp.name)
        store.documents = [
            {"content": f"doc {i}", "source": "a.pdf", "session_id": session}
            for i in range(n)
        ]
        store.index.add(vectors(n))
        store._persist()
        return store

    def test_persist_writes_both_files(self):
        self.populated()
        self.assertTrue((Path(self.tmp.name) / "index.faiss").exists())
        self.assertTrue((Path(self.tmp.name) / "documents.json").exists())

    def test_restore_recovers_vectors_and_documents(self):
        self.populated(n=4)

        revived = fake_store(self.tmp.name)
        revived._restore()

        self.assertEqual(len(revived.documents), 4)
        self.assertEqual(revived.index.ntotal, 4)
        self.assertEqual(revived.documents[0]["session_id"], "alice")

    def test_restore_is_a_noop_when_nothing_persisted(self):
        store = fake_store(self.tmp.name)
        store._restore()
        self.assertEqual(store.documents, [])

    def test_mismatched_pair_is_discarded_rather_than_trusted(self):
        """Vectors and documents out of step would misattribute every hit."""
        self.populated(n=4)
        docs = Path(self.tmp.name) / "documents.json"
        docs.write_text(json.dumps([{"content": "only one"}]), encoding="utf-8")

        revived = fake_store(self.tmp.name)
        revived._restore()

        self.assertEqual(revived.documents, [])
        self.assertEqual(revived.index.ntotal, 0)

    def test_corrupt_index_does_not_raise(self):
        self.populated()
        (Path(self.tmp.name) / "index.faiss").write_bytes(b"garbage")

        revived = fake_store(self.tmp.name)
        revived._restore()
        self.assertEqual(revived.documents, [])

    def test_no_temp_files_survive_a_write(self):
        self.populated()
        leftovers = [p.name for p in Path(self.tmp.name).iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_persistence_can_be_disabled(self):
        with patch.object(vs_mod.settings, "PERSIST_INDEX", False):
            store = fake_store(self.tmp.name)
            store.documents = [{"content": "x", "session_id": "a"}]
            store.index.add(vectors(1))
            store._persist()
        self.assertFalse((Path(self.tmp.name) / "index.faiss").exists())


class PruneTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.multiple(
            vs_mod.settings, PERSIST_INDEX=True, INDEX_DIR=self.tmp.name, EMBEDDING_DIM=DIM
        )
        p.start()
        self.addCleanup(p.stop)

    async def build(self):
        store = fake_store(self.tmp.name)
        store.documents = [
            {"content": "kb", "session_id": "kb_default"},
            {"content": "live", "session_id": "alice"},
            {"content": "dead", "session_id": "ghost"},
        ]
        store.index.add(vectors(3))
        # _build_index would need a real embedder; the rebuild is stubbed out
        store._build_index = lambda: _noop()
        return store

    async def test_expired_sessions_are_dropped(self):
        store = await self.build()
        removed = await store.prune_sessions(
            lambda sid: sid in {"kb_default", "alice"}
        )
        self.assertEqual(removed, 1)
        self.assertEqual(
            sorted(d["session_id"] for d in store.documents), ["alice", "kb_default"]
        )

    async def test_nothing_to_prune_leaves_store_untouched(self):
        store = await self.build()
        removed = await store.prune_sessions(lambda sid: True)
        self.assertEqual(removed, 0)
        self.assertEqual(len(store.documents), 3)


async def _noop():
    return None


if __name__ == "__main__":
    unittest.main()
