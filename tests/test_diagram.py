"""Mermaid validation and label repair."""

import json
import unittest

from app.main import capture_visuals
from app.services import diagram
from app.services.tools import execute_tool

# The line that aborted four consecutive flowcharts in a real session
BROKEN = """flowchart TD
    A[Agent] -->|calls| B[GET /v2/auth/agent/phone-status (phone_number, x-agent-secret)]
    B --> C{Is phone known?}
    C -->|no| E[Send signup link (signupId)]"""


class RepairTest(unittest.TestCase):
    def test_unquoted_parentheses_are_quoted(self):
        out = diagram.build(BROKEN)
        self.assertTrue(out["success"], out.get("error"))
        self.assertIn('B["GET /v2/auth/agent/phone-status (phone_number, x-agent-secret)"]', out["mermaid"])

    def test_repair_is_reported(self):
        self.assertIn("quoted", diagram.build(BROKEN)["summary"])

    def test_clean_labels_are_left_alone(self):
        src = "flowchart TD\n    A[Start] --> B[End]"
        self.assertEqual(diagram.build(src)["mermaid"], src)

    def test_inner_quotes_become_entities(self):
        out = diagram.build('flowchart TD\n    A[say "hi" (loudly)] --> B[ok]')
        self.assertIn("#quot;", out["mermaid"])
        self.assertNotIn('""', out["mermaid"])

    def test_edge_labels_are_quoted_too(self):
        out = diagram.build("flowchart TD\n    A -->|calls (twice)| B")
        self.assertTrue(out["success"], out.get("error"))
        self.assertIn('|"calls (twice)"|', out["mermaid"])

    def test_sequence_message_text_is_untouched(self):
        """Free-form text after a colon needs no quoting; quoting it would corrupt it."""
        src = "sequenceDiagram\n    Agent->>Backend: GET /x (a, b)"
        self.assertEqual(diagram.build(src)["mermaid"], src)


class ValidationTest(unittest.TestCase):
    def test_missing_diagram_type_is_rejected(self):
        out = diagram.build("A --> B")
        self.assertFalse(out["success"])
        self.assertIn("diagram type", out["error"])

    def test_supported_types_are_listed_in_the_error(self):
        self.assertIn("flowchart", diagram.build("A --> B")["error"])

    def test_empty_source_is_rejected(self):
        self.assertFalse(diagram.build("   ")["success"])

    def test_a_missing_bracket_is_reported_not_hidden(self):
        """
        A label containing an arrow means a bracket was left open. Quoting it would
        produce a valid diagram that says something the author never wrote.
        """
        out = diagram.build("flowchart TD\n    A[Start --> B[End]")
        self.assertFalse(out["success"])
        self.assertIn("line 2", out["error"])

    def test_code_fence_is_stripped(self):
        out = diagram.build("```mermaid\nflowchart LR\n    A[x] --> B[y]\n```")
        self.assertTrue(out["success"], out.get("error"))
        self.assertTrue(out["mermaid"].startswith("flowchart"))

    def test_every_supported_type_is_accepted(self):
        for kind in diagram.SUPPORTED_TYPES:
            out = diagram.build(f"{kind}\n    A --> B")
            self.assertTrue(out["success"], f"{kind}: {out.get('error')}")


class DiagramToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_returns_a_repaired_diagram(self):
        raw = await execute_tool(
            "generate_diagram", {"mermaid": BROKEN, "title": "Auth flow"}, {}
        )
        out = json.loads(raw)
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(out["title"], "Auth flow")

    async def test_tool_surfaces_a_fixable_error(self):
        raw = await execute_tool("generate_diagram", {"mermaid": "A --> B"}, {})
        self.assertFalse(json.loads(raw)["success"])


class DiagramCaptureTest(unittest.TestCase):
    """Diagrams travel out-of-band like charts, so the model never copies the source."""

    def payload(self):
        return json.dumps(
            {
                "success": True,
                "diagram_type": "flowchart",
                "title": "Auth flow",
                "mermaid": "flowchart TD\n    A --> B",
                "summary": "flowchart diagram, 2 lines",
            }
        )

    def test_diagram_is_diverted_into_the_sink(self):
        sink = []
        capture_visuals(self.payload(), sink)
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["type"], "diagram")
        self.assertEqual(sink[0]["caption"], "Auth flow")
        self.assertIn("flowchart", sink[0]["mermaid"])

    def test_receipt_withholds_the_source(self):
        sink = []
        receipt = capture_visuals(self.payload(), sink)
        self.assertNotIn("flowchart TD", receipt)
        self.assertIn("Auth flow", receipt)

    def test_failed_diagram_captures_nothing(self):
        sink = []
        receipt = capture_visuals(json.dumps({"success": False, "error": "no type"}), sink)
        self.assertEqual(sink, [])
        self.assertIn("no type", receipt)


if __name__ == "__main__":
    unittest.main()
