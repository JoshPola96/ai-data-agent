"""Request bounds: untrusted input must be rejected at the edge, not deep in a prompt."""

import json
import unittest

import pandas as pd
from pydantic import ValidationError

import app.services.tools as tools_mod
from app.core.config import get_settings
from app.services.tools import _resolve_dataframe
from app.utils.schemas import ChatRequest

settings = get_settings()


class ChatRequestLimitsTest(unittest.TestCase):
    def test_ordinary_request_is_accepted(self):
        r = ChatRequest(session_id="demo-alice_1", query="what is revenue?")
        self.assertEqual(r.session_id, "demo-alice_1")

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest(query="")

    def test_oversized_query_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest(query="x" * (settings.MAX_QUERY_CHARS + 1))

    def test_query_at_the_limit_is_accepted(self):
        ChatRequest(query="x" * settings.MAX_QUERY_CHARS)

    def test_session_id_with_redis_key_separator_is_rejected(self):
        """session_id is interpolated into Redis keys, so ':' must not pass."""
        with self.assertRaises(ValidationError):
            ChatRequest(session_id="alice:chat", query="hi")

    def test_session_id_with_glob_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest(session_id="alice*", query="hi")

    def test_session_id_with_whitespace_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest(session_id="alice bob", query="hi")

    def test_oversized_session_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest(session_id="a" * (settings.MAX_SESSION_ID_CHARS + 1), query="hi")

    def test_absent_session_id_is_allowed(self):
        self.assertIsNone(ChatRequest(query="hi").session_id)


class CustomDataLimitsTest(unittest.TestCase):
    def test_oversized_custom_data_is_refused_with_guidance(self):
        rows = [{"a": i} for i in range(settings.MAX_CUSTOM_DATA_ROWS + 1)]
        df, err = _resolve_dataframe({"custom_data": rows}, {})
        self.assertIsNone(df)
        self.assertIn("limit is", err)

    def test_custom_data_at_the_limit_is_accepted(self):
        rows = [{"a": i} for i in range(settings.MAX_CUSTOM_DATA_ROWS)]
        df, err = _resolve_dataframe({"custom_data": rows}, {})
        self.assertIsNone(err)
        self.assertEqual(len(df), settings.MAX_CUSTOM_DATA_ROWS)

    def test_empty_custom_data_falls_through_to_table(self):
        df, err = _resolve_dataframe(
            {"custom_data": [], "table_name": "t"}, {"t": pd.DataFrame({"a": [1]})}
        )
        self.assertIsNone(err)
        self.assertEqual(len(df), 1)


class DashboardLimitsTest(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_respects_the_custom_data_cap(self):
        rows = [{"a": i, "b": i} for i in range(settings.MAX_CUSTOM_DATA_ROWS + 1)]
        raw = await tools_mod.execute_tool(
            "generate_dashboard",
            {
                "custom_data": rows,
                "charts": [{"chart_type": "bar", "x_column": "a", "title": "T"}],
            },
            {},
        )
        out = json.loads(raw)
        self.assertFalse(out["success"])
        self.assertIn("limit is", out["error"])


if __name__ == "__main__":
    unittest.main()


class SessionIdRulesMatchAcrossEndpoints(unittest.TestCase):
    """
    /upload validated nothing while /chat enforced a charset, so a file could be uploaded
    under an id that could never be asked a question — and the restriction exists because
    the id is interpolated into Redis keys, which makes the unvalidated door the wrong one
    to leave open.
    """

    def test_both_endpoints_share_one_pattern(self):
        import inspect

        import app.main as main
        from app.utils.schemas import SESSION_ID_PATTERN, ChatRequest

        chat_rule = ChatRequest.model_fields["session_id"]
        self.assertTrue(
            any(getattr(m, "pattern", None) == SESSION_ID_PATTERN for m in chat_rule.metadata),
            "ChatRequest should use the shared pattern",
        )

        # Form keeps its constraints in annotated metadata rather than as attributes
        upload_rule = inspect.signature(main.upload).parameters["session_id"].default
        constraints = list(upload_rule.metadata)
        self.assertTrue(
            any(getattr(c, "pattern", None) == SESSION_ID_PATTERN for c in constraints),
            f"/upload should share the pattern; has {constraints}",
        )
        self.assertTrue(
            any(getattr(c, "max_length", None) == main.settings.MAX_SESSION_ID_CHARS
                for c in constraints),
            f"/upload should share the length bound; has {constraints}",
        )

    def test_the_pattern_rejects_what_would_break_a_key(self):
        import re

        from app.utils.schemas import SESSION_ID_PATTERN

        allowed = re.compile(SESSION_ID_PATTERN)
        for good in ("abc", "a_b-1", "A1"):
            self.assertTrue(allowed.match(good), good)
        for bad in ("has space", "dots.in.name", "colon:sep", "star*", ""):
            self.assertFalse(allowed.match(bad), bad)
