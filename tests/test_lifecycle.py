"""
A document's whole life: uploaded, re-uploaded, corrected, deleted, cleared.

The two most damaging defects in this project were both here, and neither was a bug in a
function. A re-upload duplicated every chunk because identity came from the click rather
than the bytes. A delete matched on filename alone, so removing one session's report
removed another's. A third — a corrupt re-upload destroying the good copy it was meant to
replace — was found by tabulating this grid rather than by reading the code again.

The stores are real. Redis and the embedder are stood in for, so the whole life of a
document runs in milliseconds and in the offline suite, where it will actually be run.
"""

import io
import unittest

import pandas as pd

from app.services.ingest import IngestionService
from tests.test_stores import session_store, vector_store


def workbook(rows, sheets=1) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for i in range(sheets):
            pd.DataFrame(rows).to_excel(writer, sheet_name=f"S{i + 1}", index=False)
    return buffer.getvalue()


ONE = [{"site": "Riyadh", "tickets": 10}, {"site": "Jeddah", "tickets": 20}]
TWO = [{"site": "Riyadh", "tickets": 99}, {"site": "Jeddah", "tickets": 20}]

# Built once and reused, because openpyxl stamps a creation time into the archive: the
# same rows written twice are not the same bytes, while a file re-sent from disk is.
OPS_ONE, OPS_TWO = workbook(ONE), workbook(TWO)
CORRUPT = b"\x00\x01\x02 not a workbook"


class LifecycleCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sessions = session_store()
        self.vectors = vector_store()
        self.svc = IngestionService.__new__(IngestionService)
        self.svc.session_store = self.sessions
        self.svc.vector_store = self.vectors

    async def upload(self, session, name, data):
        return await self.svc.ingest_file(session, name, data)

    async def filenames(self, session):
        docs = await self.sessions.get_session_documents(session)
        return sorted(d["filename"] for d in docs)

    async def tables(self, session):
        return sorted(await self.sessions.list_dataframes(session))

    def indexed(self):
        return self.vectors.index.ntotal


class AFirstUpload(LifecycleCase):
    async def test_it_registers_once_with_its_table_and_vectors(self):
        result = await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(result["status"], "success")
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])
        self.assertEqual(len(await self.tables("alice")), 1)
        self.assertGreater(self.indexed(), 0)

    async def test_a_file_nothing_can_be_read_from_is_refused(self):
        """It used to report success, then sit in the sidebar looking ingested."""
        with self.assertRaises(ValueError):
            await self.upload("alice", "ops.xlsx", CORRUPT)
        self.assertEqual(await self.filenames("alice"), [])

    async def test_an_unsupported_extension_is_refused(self):
        with self.assertRaises(ValueError):
            await self.upload("alice", "notes.md", b"# heading\n\nprose\n")


class AnIdenticalReUpload(LifecycleCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.before = self.indexed()

    async def test_it_is_reported_as_unchanged(self):
        result = await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(result["status"], "unchanged")

    async def test_it_adds_no_second_registry_entry(self):
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])

    async def test_it_adds_no_second_copy_of_the_chunks(self):
        """Two copies compete for the same top-k slots and crowd out other documents."""
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(self.indexed(), self.before)


class ACorrectedReUpload(LifecycleCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.before = self.indexed()

    async def test_it_supersedes_rather_than_accumulates(self):
        result = await self.upload("alice", "ops.xlsx", OPS_TWO)
        self.assertEqual(result["status"], "success")
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])
        self.assertEqual(self.indexed(), self.before)

    async def test_the_superseded_figures_are_gone(self):
        await self.upload("alice", "ops.xlsx", OPS_TWO)
        table = await self.sessions.get_dataframe(
            "alice", (await self.tables("alice"))[0]
        )
        self.assertIn(99, table["tickets"].tolist())
        self.assertNotIn(10, table["tickets"].tolist())

    async def test_sheets_the_new_version_dropped_are_not_left_behind(self):
        await self.upload("alice", "multi.xlsx", workbook(ONE, sheets=3))
        self.assertEqual(len([t for t in await self.tables("alice") if "multi" in t]), 3)
        await self.upload("alice", "multi.xlsx", workbook(TWO, sheets=1))
        self.assertEqual(len([t for t in await self.tables("alice") if "multi" in t]), 1)

    async def test_a_corrupt_replacement_does_not_destroy_the_good_copy(self):
        """
        Superseding ran before parsing, so uploading a corrupt copy of a good file
        deleted the good one and then failed — the upload run to refresh the data
        destroyed it, and the error message said only that the new file was unreadable.
        """
        with self.assertRaises(ValueError):
            await self.upload("alice", "ops.xlsx", CORRUPT)

        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])
        self.assertEqual(len(await self.tables("alice")), 1)
        self.assertEqual(self.indexed(), self.before)

    async def test_the_surviving_copy_still_holds_its_data(self):
        with self.assertRaises(ValueError):
            await self.upload("alice", "ops.xlsx", CORRUPT)
        table = await self.sessions.get_dataframe(
            "alice", (await self.tables("alice"))[0]
        )
        self.assertEqual(table["tickets"].sum(), 30)


class DeletingOneDocument(LifecycleCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        await self.upload("alice", "other.csv", b"region,revenue\nNorth,100\n")
        docs = await self.sessions.get_session_documents("alice")
        self.target = next(d for d in docs if d["filename"] == "other.csv")

    async def delete_target(self):
        await self.sessions.delete_document("alice", self.target["doc_id"], self.vectors)

    async def test_it_leaves_the_registry(self):
        await self.delete_target()
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])

    async def test_its_table_stops_being_queryable(self):
        """A deleted spreadsheet that still answers questions is worse than a stale one."""
        await self.delete_target()
        self.assertEqual([t for t in await self.tables("alice") if "other" in t], [])

    async def test_the_other_documents_tables_survive(self):
        await self.delete_target()
        self.assertEqual(len([t for t in await self.tables("alice") if "ops" in t]), 1)

    async def test_deleting_it_twice_is_harmless(self):
        await self.delete_target()
        await self.delete_target()
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])

    async def test_deleting_a_document_that_was_never_here_is_harmless(self):
        await self.sessions.delete_document("alice", "no-such-doc", self.vectors)
        self.assertEqual(len(await self.filenames("alice")), 2)


class OneSessionCannotReachAnother(LifecycleCase):
    """The same filename in two sessions is two documents, and was once treated as one."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.upload("alice", "report.xlsx", OPS_ONE)
        await self.upload("bob", "report.xlsx", OPS_ONE)
        docs = await self.sessions.get_session_documents("alice")
        self.alice_doc = docs[0]["doc_id"]

    async def test_both_sessions_hold_their_own_copy(self):
        self.assertEqual(await self.filenames("alice"), ["report.xlsx"])
        self.assertEqual(await self.filenames("bob"), ["report.xlsx"])

    async def test_deleting_one_leaves_the_others_registry(self):
        await self.sessions.delete_document("alice", self.alice_doc, self.vectors)
        self.assertEqual(await self.filenames("alice"), [])
        self.assertEqual(await self.filenames("bob"), ["report.xlsx"])

    async def test_deleting_one_leaves_the_others_tables(self):
        await self.sessions.delete_document("alice", self.alice_doc, self.vectors)
        self.assertEqual(len(await self.tables("bob")), 1)

    async def test_deleting_one_leaves_the_others_vectors(self):
        before = self.indexed()
        await self.sessions.delete_document("alice", self.alice_doc, self.vectors)
        self.assertGreater(self.indexed(), 0, "bob's chunks went with alice's")
        self.assertLess(self.indexed(), before)

    async def test_clearing_one_session_leaves_the_other_whole(self):
        await self.sessions.clear_session("alice", self.vectors)
        self.assertEqual(await self.filenames("alice"), [])
        self.assertEqual(await self.tables("alice"), [])
        self.assertEqual(await self.filenames("bob"), ["report.xlsx"])
        self.assertEqual(len(await self.tables("bob")), 1)

    async def test_clearing_an_empty_session_is_harmless(self):
        await self.sessions.clear_session("carol", self.vectors)
        self.assertEqual(await self.filenames("bob"), ["report.xlsx"])


class TheRegistryAndTheIndexAgree(LifecycleCase):
    """
    Either half drifting is invisible: an orphaned vector answers for a document the
    sidebar no longer lists, and an orphaned entry lists a document with nothing behind it.
    """

    async def test_no_vectors_are_left_once_every_document_is_deleted(self):
        await self.upload("alice", "a.xlsx", OPS_ONE)
        await self.upload("alice", "b.csv", b"region,revenue\nNorth,100\n")
        for doc in await self.sessions.get_session_documents("alice"):
            await self.sessions.delete_document("alice", doc["doc_id"], self.vectors)

        self.assertEqual(await self.filenames("alice"), [])
        self.assertEqual(await self.tables("alice"), [])
        self.assertEqual(self.indexed(), 0)

    async def test_every_registered_table_can_actually_be_read(self):
        await self.upload("alice", "multi.xlsx", workbook(ONE, sheets=3))
        docs = await self.sessions.get_session_documents("alice")
        for name in docs[0]["tables"]:
            self.assertIsNotNone(
                await self.sessions.get_dataframe("alice", name),
                f"'{name}' is registered but unreadable",
            )

    async def test_the_recorded_table_count_matches_the_tables_stored(self):
        await self.upload("alice", "multi.xlsx", workbook(ONE, sheets=3))
        docs = await self.sessions.get_session_documents("alice")
        self.assertEqual(docs[0]["dataframes"], len(await self.tables("alice")))


class CleaningRunsOnStringDtypeColumns(unittest.IsolatedAsyncioTestCase):
    """
    pandas is midway through moving text out of the "object" dtype. Every cleaning pass
    selected on "object" alone, which still matches today only for backward
    compatibility — so on the release that drops it, dates would stay strings, currency
    would stay text and whitespace variants would stop collapsing, silently and without
    a single failing assertion anywhere else in this suite.
    """

    async def asyncSetUp(self):
        self.svc = IngestionService.__new__(IngestionService)

    async def clean(self, frame):
        return await self.svc._universal_data_cleaning(frame, "t")

    async def test_currency_text_is_converted_when_the_column_is_string_dtype(self):
        frame = pd.DataFrame({"amount": pd.array(["1,200.50", "2,300.75"], dtype="string")})
        self.assertAlmostEqual((await self.clean(frame))["amount"].sum(), 3501.25, places=2)

    async def test_whitespace_is_stripped_when_the_column_is_string_dtype(self):
        frame = pd.DataFrame({"region": pd.array([" North ", "South  "], dtype="string")})
        self.assertEqual(
            sorted((await self.clean(frame))["region"].tolist()), ["North", "South"]
        )

    async def test_the_object_dtype_path_still_behaves_the_same(self):
        frame = pd.DataFrame({"amount": ["1,200.50", "2,300.75"]})
        self.assertAlmostEqual((await self.clean(frame))["amount"].sum(), 3501.25, places=2)


class ADiscardedIndexCanBeRebuilt(LifecycleCase):
    """
    The index and the registry can part company: a changed embedding model, a torn write
    or a lost volume all end with the index discarded while Redis keeps every entry.

    Every file then looked byte-identical to something already present and was skipped,
    so the corpus came up empty and stayed empty — re-ingesting was precisely what the
    skip prevented, and the startup log said "unchanged" as if all were well.
    """

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertGreater(self.indexed(), 0)

    async def lose_the_index(self):
        await self.vectors.clear()
        self.assertEqual(self.indexed(), 0)

    async def test_a_registered_but_unindexed_file_is_ingested_again(self):
        await self.lose_the_index()
        result = await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(result["status"], "success")
        self.assertGreater(self.indexed(), 0, "the document is still not searchable")

    async def test_it_does_not_leave_two_registry_entries_behind(self):
        await self.lose_the_index()
        await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(await self.filenames("alice"), ["ops.xlsx"])

    async def test_an_indexed_file_is_still_skipped(self):
        """The repair must not cost the deduplication it was built around."""
        result = await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(result["status"], "unchanged")

    async def test_another_sessions_copy_does_not_count_as_this_ones(self):
        await self.upload("bob", "ops.xlsx", OPS_ONE)
        await self.vectors.remove_by_session("alice")
        result = await self.upload("alice", "ops.xlsx", OPS_ONE)
        self.assertEqual(result["status"], "success", "bob's vectors passed for alice's")


class TheSuiteNeverWritesTheRealIndex(unittest.TestCase):
    """
    A test store used to persist over the configured index, which is how the knowledge
    base was discovered empty. Running the suite must never cost a running instance its
    corpus.
    """

    def test_a_test_store_has_persistence_disabled(self):
        self.assertFalse(vector_store().persist)

    def test_the_flag_is_read_once_rather_than_at_every_write(self):
        """Reading settings inside _persist would ignore a per-instance override."""
        import inspect

        from app.core.vectorstore import VectorStore

        source = inspect.getsource(VectorStore._persist) + inspect.getsource(
            VectorStore._restore
        )
        self.assertNotIn("settings.PERSIST_INDEX", source)
        self.assertIn("self.persist", source)


class CleaningNeverConsumesTheOriginal(unittest.IsolatedAsyncioTestCase):
    """
    A payslip exported as "Basic Pay 5000" hides a figure nothing can sum, so the number
    is lifted into its own column. That step used to *replace* the original with the text
    part — and "Widget 500" and "Widget 750" both became "Widget". Two products collapsed
    into one identity, so every grouping double-counted them, silently.

    "Basic Pay 5000" and "Widget 500" are the same shape; no signal separates a merged
    column from a product name. So the split is additive: the original is never touched,
    and being wrong costs a spare column instead of a row's identity.
    """

    async def clean(self, frame):
        svc = IngestionService.__new__(IngestionService)
        return await svc._universal_data_cleaning(frame, "t")

    async def test_a_product_name_keeps_its_number(self):
        frame = pd.DataFrame({"item": ["Widget 500", "Widget 750", "Gadget 20"],
                              "n": [1, 2, 3]})
        cleaned = await self.clean(frame)
        self.assertEqual(cleaned["item"].tolist(),
                         ["Widget 500", "Widget 750", "Gadget 20"])

    async def test_two_products_do_not_collapse_into_one(self):
        """The failure that mattered: a groupby would have summed them together."""
        frame = pd.DataFrame({"item": ["Widget 500", "Widget 750"], "n": [1, 2]})
        cleaned = await self.clean(frame)
        self.assertEqual(cleaned["item"].nunique(), 2)

    async def test_the_embedded_figure_is_still_made_summable(self):
        frame = pd.DataFrame({"desc": ["Basic Pay 5000", "Allowance 300",
                                       "Deduction 100"], "n": [1, 2, 3]})
        cleaned = await self.clean(frame)
        self.assertEqual(cleaned["desc_Value"].sum(), 5400)
        self.assertEqual(cleaned["desc"].iloc[0], "Basic Pay 5000")

    async def test_a_year_in_a_document_title_survives(self):
        frame = pd.DataFrame({"doc": ["Report 2024", "Report 2025"], "n": [1, 2]})
        cleaned = await self.clean(frame)
        self.assertEqual(cleaned["doc"].nunique(), 2)


class AnEmptyColumnCannotVouchForATotalsRow(unittest.IsolatedAsyncioTestCase):
    """
    The totals-row test falls back to the row's labels when only one numeric column
    exists. A column of nulls is not a label — it is an absent column — but it counted as
    a blank one, which is what a genuine totals line looks like. A table of 1, 2, 3 beside
    an empty column lost its third row and the total fell from 6 to 3.
    """

    async def test_a_table_with_an_empty_column_keeps_every_row(self):
        svc = IngestionService.__new__(IngestionService)
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "empty": [None, None, None]})
        cleaned = await svc._universal_data_cleaning(frame, "t")
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned["a"].sum(), 6.0)

    def test_the_helper_ignores_all_null_columns_when_looking_for_labels(self):
        from app.utils.helpers import totals_row_index

        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "empty": [None, None, None]})
        self.assertIsNone(totals_row_index(frame))
