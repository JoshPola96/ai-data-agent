"""Chunking: size limits, overlap carry-over, and short-document handling."""

import unittest
from unittest.mock import patch

import app.services.ingest as ingest_mod
from app.services.ingest import IngestionService

# Distinct tokens per paragraph, so shared tokens can only come from carry-over
PARAS = ["p%d " % i + " ".join(f"t{i}w{j}" for j in range(20)) for i in range(6)]
TEXT = "\n\n".join(PARAS)


def boundaries_sharing_tokens(chunks):
    return sum(
        1
        for a, b in zip(chunks, chunks[1:])
        if set(a["content"].split()) & set(b["content"].split())
    )


class ChunkingTest(unittest.TestCase):
    def setUp(self):
        self.svc = IngestionService(None, None)
        self.settings_patch = patch.multiple(
            ingest_mod.settings,
            CHUNK_SIZE=200,
            CHUNK_OVERLAP=40,
            MIN_CHUNK_SIZE=100,
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def chunk(self, text):
        return self.svc._semantic_chunking(text, "s.txt", "doc1")

    def test_short_document_still_produces_a_chunk(self):
        self.assertEqual(len(self.chunk("Tiny note about revenue.")), 1)

    def test_empty_text_produces_nothing(self):
        self.assertEqual(self.chunk("   \n\n  "), [])

    def test_long_text_splits(self):
        self.assertGreater(len(self.chunk(TEXT)), 1)

    def test_every_boundary_carries_context_forward(self):
        chunks = self.chunk(TEXT)
        self.assertEqual(boundaries_sharing_tokens(chunks), len(chunks) - 1)

    def test_zero_overlap_disables_carry_over(self):
        with patch.object(ingest_mod.settings, "CHUNK_OVERLAP", 0):
            chunks = self.chunk(TEXT)
        self.assertEqual(boundaries_sharing_tokens(chunks), 0)

    def test_overlap_starts_on_a_word_boundary(self):
        with patch.object(ingest_mod.settings, "CHUNK_OVERLAP", 25):
            chunks = self.chunk(TEXT)
        for c in chunks[1:]:
            first = c["content"].split()[0]
            self.assertTrue(
                first.startswith("t") or first.startswith("p"),
                f"chunk begins mid-token: {first!r}",
            )

    def test_chunks_carry_source_metadata(self):
        for c in self.chunk(TEXT):
            self.assertEqual(c["source"], "s.txt")
            self.assertEqual(c["doc_id"], "doc1")
            self.assertEqual(c["metadata"]["type"], "text")


if __name__ == "__main__":
    unittest.main()
