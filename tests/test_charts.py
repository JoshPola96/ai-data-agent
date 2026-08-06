"""Dashboard generation and out-of-band chart capture."""

import json
import unittest

import pandas as pd

from app.main import capture_charts
from app.services.chart import ChartService
from app.services.tools import execute_tool, _resolve_dataframe

DF = pd.DataFrame(
    {
        "region": ["North"] * 4 + ["South"] * 4 + ["East"] * 4 + ["West"] * 4,
        "quarter": ["Q1", "Q2", "Q3", "Q4"] * 4,
        "revenue": [120, 135, 128, 171, 98, 104, 99, 142, 156, 161, 149, 203, 87, 91, 88, 119],
        "units": [340, 390, 355, 470, 290, 310, 295, 430, 410, 425, 395, 540, 260, 275, 265, 360],
        "margin": [18.5, 19.2, 17.8, 21.4, 15.1, 15.9, 14.7, 19.8, 22.3, 22.9, 21.1, 25.6, 13.4, 13.9, 12.8, 17.2],
    }
)

SPECS = [
    {"chart_type": "bar", "x_column": "region", "y_column": "revenue", "aggregation": "sum", "title": "By Region"},
    {"chart_type": "line", "x_column": "quarter", "y_column": "revenue", "aggregation": "sum", "title": "Trend"},
    {"chart_type": "scatter", "x_column": "units", "y_column": "revenue", "title": "Units vs Revenue"},
]


class DashboardTest(unittest.IsolatedAsyncioTestCase):
    async def run_tool(self, charts, **extra):
        raw = await execute_tool(
            "generate_dashboard", {"table_name": "sales", "charts": charts, **extra}, {"sales": DF}
        )
        return json.loads(raw)

    async def test_builds_every_valid_chart(self):
        out = await self.run_tool(SPECS)
        self.assertTrue(out["success"])
        self.assertEqual(len(out["charts"]), 3)
        self.assertEqual(out["failures"], [])

    async def test_one_bad_spec_does_not_lose_the_others(self):
        bad = {"chart_type": "bar", "x_column": "NOPE", "title": "Broken"}
        out = await self.run_tool(SPECS + [bad])
        self.assertEqual(len(out["charts"]), 3)
        self.assertEqual(len(out["failures"]), 1)
        self.assertEqual(out["failures"][0]["title"], "Broken")
        self.assertTrue(out["success"], "partial success must still be success")

    async def test_success_is_false_when_nothing_builds(self):
        out = await self.run_tool([{"chart_type": "bar", "x_column": "NOPE", "title": "Broken"}])
        self.assertFalse(out["success"])

    async def test_chart_count_is_capped(self):
        out = await self.run_tool(SPECS * 4)
        self.assertLessEqual(len(out["charts"]), ChartService.MAX_DASHBOARD_CHARTS)

    async def test_missing_specs_is_an_error_not_a_crash(self):
        out = await self.run_tool([])
        self.assertFalse(out["success"])
        self.assertIn("No chart specifications", out["error"])

    async def test_unknown_table_reports_available_tables(self):
        raw = await execute_tool(
            "generate_dashboard", {"table_name": "ghost", "charts": SPECS}, {"sales": DF}
        )
        out = json.loads(raw)
        self.assertFalse(out["success"])
        self.assertIn("sales", out["error"])


class ColorColumnTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: aggregating dropped the colour column.

    groupby(x)[y].reset_index() keeps only those two columns, so plotting then failed
    on a colour column that no longer existed — breaking every "trend over time, split
    by category" request, which is one of the most natural things to ask for.
    """

    async def chart(self, **args):
        raw = await execute_tool("generate_chart", {"table_name": "sales", **args}, {"sales": DF})
        return json.loads(raw)

    async def test_line_split_by_category_survives_aggregation(self):
        out = await self.chart(
            chart_type="line", x_column="quarter", y_column="revenue",
            aggregation="sum", color_column="region", title="Trend by region",
        )
        self.assertTrue(out["success"], out.get("error"))
        names = [t.get("name") for t in out["chart_json"]["data"]]
        self.assertEqual(sorted(names), ["East", "North", "South", "West"])

    async def test_count_aggregation_keeps_the_colour_split(self):
        out = await self.chart(
            chart_type="bar", x_column="region", aggregation="count",
            color_column="quarter", title="Counts",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(len(out["chart_json"]["data"]), 4)

    async def test_colour_column_is_fuzzy_matched_like_x_and_y(self):
        out = await self.chart(
            chart_type="line", x_column="quarter", y_column="revenue",
            aggregation="sum", color_column="REGION ", title="Fuzzy",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(len(out["chart_json"]["data"]), 4)

    async def test_without_a_colour_column_a_single_series_is_produced(self):
        out = await self.chart(
            chart_type="line", x_column="quarter", y_column="revenue",
            aggregation="sum", title="Plain",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(len(out["chart_json"]["data"]), 1)

    async def test_grouped_bars_are_side_by_side_not_stacked(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", color_column="quarter", title="Grouped",
        )
        self.assertEqual(out["chart_json"]["layout"].get("barmode"), "group")


class NewChartTypesTest(unittest.IsolatedAsyncioTestCase):
    async def chart(self, **args):
        raw = await execute_tool("generate_chart", {"table_name": "sales", **args}, {"sales": DF})
        return json.loads(raw)

    async def test_box_plot(self):
        out = await self.chart(chart_type="box", x_column="region", y_column="revenue", title="Spread")
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(out["chart_json"]["data"][0]["type"], "box")

    async def test_heatmap_needs_no_x_column(self):
        out = await self.chart(chart_type="heatmap", title="Correlation")
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(out["chart_json"]["data"][0]["type"], "heatmap")

    async def test_heatmap_without_numeric_columns_fails_cleanly(self):
        raw = await execute_tool(
            "generate_chart",
            {"custom_data": [{"a": "x"}, {"a": "y"}], "chart_type": "heatmap", "title": "T"},
            {},
        )
        out = json.loads(raw)
        self.assertFalse(out["success"])
        self.assertIn("numeric", out["error"])

    async def test_top_n_trims_to_a_ranking(self):
        out = await self.chart(
            chart_type="bar", x_column="quarter", y_column="revenue",
            aggregation="sum", top_n=2, title="Top 2",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(len(out["chart_json"]["data"][0]["x"]), 2)


class CaptureChartsTest(unittest.TestCase):
    """Chart payloads must reach the response without passing through the model."""

    def payload(self, n=3):
        charts = [
            {
                "title": f"Chart {i}",
                "chart_json": {"data": [{"type": "bar", "x": ["a"], "y": [1]}], "layout": {}},
                "chart_html": "<html>" + "x" * 5000 + "</html>",
                "summary": f"summary {i}",
            }
            for i in range(n)
        ]
        return json.dumps({"success": True, "charts": charts, "failures": []})

    def test_charts_are_diverted_into_the_sink(self):
        sink = []
        capture_charts(self.payload(3), sink)
        self.assertEqual(len(sink), 3)
        self.assertTrue(all(v["type"] == "chart" for v in sink))
        self.assertTrue(all(v["chart_data"]["chart_json"] for v in sink))

    def test_receipt_excludes_the_figure_payload(self):
        sink = []
        raw = self.payload(3)
        receipt = capture_charts(raw, sink)
        self.assertNotIn("chart_html", receipt)
        self.assertNotIn("<html>", receipt)
        self.assertLess(len(receipt), len(raw) / 10, "receipt must be far smaller than the payload")

    def test_single_chart_payload_is_handled(self):
        sink = []
        single = json.dumps(
            {"success": True, "chart_json": {"data": [], "layout": {"title": {"text": "Solo"}}}, "summary": "s"}
        )
        capture_charts(single, sink)
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["caption"], "Solo")

    def test_failed_tool_output_captures_nothing(self):
        sink = []
        receipt = capture_charts(json.dumps({"success": False, "error": "bad column"}), sink)
        self.assertEqual(sink, [])
        self.assertIn("bad column", receipt)

    def test_non_json_output_does_not_raise(self):
        sink = []
        self.assertIsInstance(capture_charts("not json at all", sink), str)
        self.assertEqual(sink, [])


class ResolveDataframeTest(unittest.TestCase):
    def test_table_lookup(self):
        df, err = _resolve_dataframe({"table_name": "sales"}, {"sales": DF})
        self.assertIsNone(err)
        self.assertEqual(len(df), 16)

    def test_custom_data_wins_over_table(self):
        df, err = _resolve_dataframe(
            {"table_name": "sales", "custom_data": [{"a": 1}, {"a": 2}]}, {"sales": DF}
        )
        self.assertIsNone(err)
        self.assertEqual(list(df.columns), ["a"])

    def test_no_source_is_an_error(self):
        df, err = _resolve_dataframe({}, {})
        self.assertIsNone(df)
        self.assertIn("Must provide", err)

    def test_placeholder_table_is_not_a_source(self):
        df, err = _resolve_dataframe({"table_name": "__none__"}, {"sales": DF})
        self.assertIsNone(df)


if __name__ == "__main__":
    unittest.main()
