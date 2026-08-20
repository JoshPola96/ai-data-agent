"""Reference lines, secondary axes and time resampling."""

import base64
import json
import struct
import unittest

import pandas as pd

from app.services.chart import ChartService
from app.services.tools import execute_tool

DAILY = pd.DataFrame(
    {
        "day": pd.date_range("2025-01-01", periods=90, freq="D").astype(str),
        "revenue": [100 + (i % 7) * 10 for i in range(90)],
        "margin_pct": [12 + (i % 5) for i in range(90)],
        "region": ["North", "South"] * 45,
    }
)


def values(arr):
    """Plotly encodes numeric arrays as base64 typed-arrays, not JSON lists."""
    if isinstance(arr, dict) and "bdata" in arr:
        fmt = {"i1": "b", "i2": "h", "i4": "i", "i8": "q", "u1": "B", "f4": "f", "f8": "d"}[arr["dtype"]]
        raw = base64.b64decode(arr["bdata"])
        return list(struct.unpack("<" + fmt * (len(raw) // struct.calcsize(fmt)), raw))
    return list(arr)


class ChartOptionTest(unittest.IsolatedAsyncioTestCase):
    async def chart(self, **args):
        raw = await execute_tool("generate_chart", {"table_name": "d", **args}, {"d": DAILY})
        return json.loads(raw)


class ReferenceLineTest(ChartOptionTest):
    """A comparison without a baseline leaves "compared with what" unanswered."""

    async def test_mean_line_is_drawn(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", reference="mean", title="With mean",
        )
        self.assertTrue(out["success"], out.get("error"))
        shapes = out["chart_json"]["layout"].get("shapes", [])
        self.assertTrue(any(s.get("type") == "line" for s in shapes))

    async def test_median_is_accepted(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", reference="median", title="With median",
        )
        self.assertTrue(out["chart_json"]["layout"].get("shapes"))

    async def test_a_literal_target_is_accepted(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", reference="5000", title="Target",
        )
        self.assertTrue(out["chart_json"]["layout"].get("shapes"))

    async def test_nonsense_reference_is_ignored_not_fatal(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", reference="banana", title="Bad",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertFalse(out["chart_json"]["layout"].get("shapes"))

    async def test_no_reference_draws_no_line(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", title="Plain",
        )
        self.assertFalse(out["chart_json"]["layout"].get("shapes"))


class SecondaryAxisTest(ChartOptionTest):
    """Revenue and a percentage belong on one chart but not one scale."""

    async def test_second_measure_gets_its_own_axis(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="mean", title="Dual",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertIn("yaxis2", out["chart_json"]["layout"])

    async def test_the_second_series_is_actually_plotted(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="mean", title="Dual",
        )
        names = [t.get("name") for t in out["chart_json"]["data"]]
        self.assertIn("margin_pct", names)

    async def test_second_measure_survives_aggregation(self):
        """Aggregating only the primary y would drop the column the axis needs."""
        out = await self.chart(
            chart_type="line", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="sum", title="Dual",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertIn("yaxis2", out["chart_json"]["layout"])

    async def test_a_secondary_percentage_is_averaged_not_summed(self):
        """
        Observed live: revenue and margin_pct by region put ~990 on the right-hand axis,
        because the primary `sum` was applied to the percentage as well. margin_pct is
        12-16 here, so any region total above 20 means it was summed.
        """
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="sum",
        )
        second = [t for t in out["chart_json"]["data"] if t.get("yaxis") == "y2"][0]
        ys = values(second["y"])
        self.assertTrue(all(10 < v < 20 for v in ys), f"y2 not averaged: {ys}")

    async def test_a_summed_secondary_axis_is_still_available(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="sum", y2_aggregation="sum",
        )
        second = [t for t in out["chart_json"]["data"] if t.get("yaxis") == "y2"][0]
        self.assertTrue(max(values(second["y"])) > 100, "explicit sum was overridden")

    async def test_unknown_second_column_is_ignored(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            y2_column="nonexistent", aggregation="sum", title="Dual",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertNotIn("yaxis2", out["chart_json"]["layout"])

    async def test_secondary_axis_is_skipped_for_unsuitable_types(self):
        out = await self.chart(
            chart_type="pie", x_column="region", y_column="revenue",
            y2_column="margin_pct", aggregation="sum", title="Pie",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertNotIn("yaxis2", out["chart_json"]["layout"])


class ResampleTest(ChartOptionTest):
    """Ninety daily rows do not answer a monthly question."""

    async def test_daily_rows_collapse_into_months(self):
        out = await self.chart(
            chart_type="line", x_column="day", y_column="revenue",
            aggregation="sum", resample="M", title="Monthly",
        )
        self.assertTrue(out["success"], out.get("error"))
        points = len(out["chart_json"]["data"][0]["x"])
        self.assertEqual(points, 3, f"expected 3 months, got {points}")

    async def test_quarterly_collapses_further(self):
        out = await self.chart(
            chart_type="line", x_column="day", y_column="revenue",
            aggregation="sum", resample="Q", title="Quarterly",
        )
        self.assertEqual(len(out["chart_json"]["data"][0]["x"]), 1)

    async def test_monthly_totals_are_correct(self):
        out = await self.chart(
            chart_type="bar", x_column="day", y_column="revenue",
            aggregation="sum", resample="M", title="Monthly",
        )
        ys = out["chart_json"]["data"][0]["y"]
        ys = list(ys) if isinstance(ys, list) else None
        if ys:
            expected = (
                DAILY.assign(m=pd.to_datetime(DAILY["day"]).dt.to_period("M"))
                .groupby("m")["revenue"].sum().tolist()
            )
            self.assertEqual([round(v) for v in ys], expected)

    async def test_resampling_a_non_date_axis_is_skipped_not_fatal(self):
        out = await self.chart(
            chart_type="bar", x_column="region", y_column="revenue",
            aggregation="sum", resample="M", title="Not a date",
        )
        self.assertTrue(out["success"], out.get("error"))
        self.assertEqual(len(out["chart_json"]["data"][0]["x"]), 2)

    async def test_resample_reaches_the_dashboard_tool(self):
        raw = await execute_tool(
            "generate_dashboard",
            {"table_name": "d", "charts": [
                {"chart_type": "line", "x_column": "day", "y_column": "revenue",
                 "aggregation": "sum", "resample": "M", "title": "Monthly"},
            ]},
            {"d": DAILY},
        )
        out = json.loads(raw)
        self.assertTrue(out["success"], out.get("failures"))


class SpecTest(unittest.TestCase):
    """Resolution happens once, so every stage sees the same column names."""

    def test_columns_are_resolved_case_insensitively(self):
        spec = ChartService.resolve(
            DAILY, chart_type="bar", x_column="REGION", y_column="Revenue",
            y2_column="MARGIN_PCT",
        )
        self.assertEqual((spec.x, spec.y, spec.y2), ("region", "revenue", "margin_pct"))

    def test_a_ratio_becomes_a_derived_column(self):
        spec = ChartService.resolve(DAILY, chart_type="bar", y_column="revenue/margin_pct")
        self.assertIsNotNone(spec.ratio)
        self.assertEqual(spec.y, "revenue_per_margin_pct")

    def test_unknown_columns_resolve_to_none(self):
        spec = ChartService.resolve(DAILY, chart_type="bar", x_column="ghost")
        self.assertIsNone(spec.x)

    def test_a_reference_label_matches_the_axis_units(self):
        """A rate axis reading 3.44% beside "mean: 0.03" names a different number."""
        rates = pd.DataFrame({"cat": ["a", "b"], "rate": [0.041, 0.021]})
        out = ChartService.generate_chart(
            rates, chart_type="bar", x_column="cat", y_column="rate", reference="mean"
        )
        text = json.dumps(out["chart_json"]["layout"])
        self.assertIn("3.10%", text)
        self.assertNotIn("mean: 0.03", text)

    def test_value_columns_are_ordered(self):
        spec = ChartService.resolve(
            DAILY, chart_type="bar", y_column="revenue", y2_column="margin_pct"
        )
        self.assertEqual(spec.value_columns, ["revenue", "margin_pct"])


if __name__ == "__main__":
    unittest.main()
