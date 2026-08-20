# app/services/chart.py

"""
Plotly Chart Service - Production Ready with Enhanced Visualizations
Supports multiple charts, responsive sizing, and rich metadata
"""

import logging
import warnings
from dataclasses import dataclass
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
from difflib import get_close_matches
from app.utils.helpers import canonical_labels, to_numeric
import json

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Parsing dates.*dayfirst.*")
warnings.filterwarnings("ignore", message=".*Could not infer format.*")


@dataclass
class ChartSpec:
    """
    One chart request with every column already resolved against the dataframe.

    Resolution happens once, up front, so each stage of the pipeline sees the same
    names. Threading a dozen positional arguments through preparation, plotting and
    annotation is how those stages drift out of agreement.
    """

    chart_type: str
    title: str = "Chart"
    x: Optional[str] = None
    y: Optional[str] = None
    y2: Optional[str] = None
    color: Optional[str] = None
    aggregation: str = "none"
    y2_aggregation: str = "mean"
    normalize_keys: bool = False
    top_n: Optional[int] = None
    resample: Optional[str] = None
    reference: Optional[str] = None
    ratio: Optional[Dict[str, str]] = None
    # How many groups existed before top_n trimmed them, when it did. A one-bar chart
    # beside prose naming four regions is a contradiction the reader sees and the model
    # does not, so the receipt has to say what was left out.
    trimmed_from: Optional[int] = None

    @property
    def value_columns(self) -> List[str]:
        """Columns carrying measurements, in axis order."""
        return [c for c in (self.y, self.y2) if c]


class ChartService:
    """Enhanced chart generation service with Plotly"""

    # Python equivalents of the d3 axis formats _value_format returns, so a reference
    # annotation is written in the same units as the axis it labels.
    _ANNOTATION_FORMATS = {".2%": "{:.2%}", ".3s": "{:,.0f}", ".4g": "{:,.4g}"}

    # Chart configurations for optimal rendering
    CHART_CONFIG = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
    }

    # Default layout settings
    DEFAULT_LAYOUT = {
        "template": "plotly_white",
        "height": 450,  # Better fit for dashboards
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
        "font": {"family": "Arial, sans-serif", "size": 12},
        "hovermode": "closest",
    }

    @staticmethod
    def _fuzzy_col_match(df: pd.DataFrame, target: str) -> Optional[str]:
        """Fuzzy column matching with case-insensitive search and type safety"""
        if not target:
            return None

        # Ensure all column names are strings
        df.columns = df.columns.astype(str)

        if target in df.columns:
            return target

        # Case-insensitive exact match
        cols = {str(c).lower().strip(): str(c) for c in df.columns}
        target_clean = str(target).lower().strip()

        if target_clean in cols:
            return cols[target_clean]

        # Fuzzy match
        matches = get_close_matches(target_clean, cols.keys(), n=1, cutoff=0.6)
        if matches:
            matched = cols[matches[0]]
            logger.info(f"📊 Fuzzy matched '{target}' → '{matched}'")
            return matched

        logger.warning(f"⚠️  Column '{target}' not found in {list(df.columns)}")
        return None

    @staticmethod
    def resolve_ratio(df: pd.DataFrame, spec: str) -> Optional[Dict[str, str]]:
        """
        Materialise "returns/units" as a real column so rates can be plotted.

        Rates, shares and per-unit values are the questions people actually ask, and
        none of them exist as a column in the source data. Only a division of two
        existing numeric columns is accepted — no expressions, no eval.

        Returns the derived name alongside its source columns, because aggregating a
        rate correctly means summing both sides and dividing once, not averaging the
        per-row ratios.
        """
        if not spec or "/" not in spec:
            return None

        num, _, den = spec.partition("/")
        num = ChartService._fuzzy_col_match(df, num.strip())
        den = ChartService._fuzzy_col_match(df, den.strip())
        if not num or not den:
            return None

        name = f"{num}_per_{den}"
        df[num] = to_numeric(df[num])
        df[den] = to_numeric(df[den])
        df[name] = df[num] / df[den].replace(0, np.nan)

        logger.info(f"  ➗ Derived '{name}' from {num}/{den}")
        return {"name": name, "num": num, "den": den}

    @staticmethod
    def resolve(df: pd.DataFrame, **request) -> ChartSpec:
        """Map a tool request onto columns that exist, deriving any ratio first."""
        y_column = request.get("y_column")
        ratio = ChartService.resolve_ratio(df, y_column) if y_column else None

        def col(name):
            return ChartService._fuzzy_col_match(df, name) if name else None

        return ChartSpec(
            chart_type=request.get("chart_type", "bar"),
            title=request.get("title", "Chart"),
            x=col(request.get("x_column")),
            y=ratio["name"] if ratio else col(y_column),
            y2=col(request.get("y2_column")),
            color=col(request.get("color_column")),
            aggregation=request.get("aggregation", "none"),
            y2_aggregation=request.get("y2_aggregation", "mean"),
            normalize_keys=bool(request.get("normalize_keys")),
            top_n=request.get("top_n"),
            resample=request.get("resample"),
            reference=request.get("reference"),
            ratio=ratio,
        )

    @staticmethod
    def bucket_dates(df: pd.DataFrame, column: str, freq: str) -> pd.DataFrame:
        """
        Collapse a date column to period starts, so daily rows can answer a monthly
        question.

        Shared with the statistics tool rather than reimplemented there: only the chart
        could bucket, so "which month was highest" had to be read off the picture, and a
        figure read off a picture is a figure nobody checked.
        """
        dates = pd.to_datetime(df[column], errors="coerce")
        if dates.notna().mean() <= 0.7:
            logger.warning(f"  ⚠️ '{column}' is not a date column, resample skipped")
            return df

        bucketed = df.assign(
            **{column: dates.dt.to_period(freq).dt.to_timestamp()}
        ).dropna(subset=[column])
        logger.info(f"  🗓 Resampled {column} to '{freq}' buckets")
        return bucketed

    @staticmethod
    def _clean_and_sort_data(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
        """Clean and sort data intelligently"""
        plot_df = df.copy()

        # Try date parsing and sorting
        try:
            plot_df["_sort_temp"] = pd.to_datetime(plot_df[x_col], errors="coerce")

            # If >50% are valid dates, sort by them
            if plot_df["_sort_temp"].notna().sum() > len(plot_df) * 0.5:
                plot_df = plot_df.sort_values("_sort_temp")
                logger.info(f"  📅 Sorted by date column: {x_col}")

            plot_df = plot_df.drop(columns=["_sort_temp"], errors="ignore")
        except Exception as e:
            logger.debug(f"  ℹ️  Date parsing skipped: {e}")

        return plot_df

    @staticmethod
    def _prepare_data(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        """Clean, optionally resample the time axis, aggregate, then trim to a ranking."""
        x, y = spec.x, spec.y

        # Four regions written nine ways chart as nine bars, and every total is split
        # between the spellings. Collapsing them is opt-in: elsewhere two values that
        # differ only in case are genuinely two values.
        if spec.normalize_keys:
            for col in (x, spec.color):
                if col and col in df.columns and df[col].dtype.kind not in "ifbcmM":
                    before = df[col].nunique()
                    df[col] = canonical_labels(df[col])
                    after = df[col].nunique()
                    if after < before:
                        logger.info(f"  ⇢ Normalised '{col}': {before} labels to {after}")

        # Categorical charts need string x-axis
        if spec.chart_type in ("bar", "pie") and not spec.resample:
            df[x] = df[x].astype(str).fillna("Unknown")

        for col in spec.value_columns:
            if col != "count" and col in df.columns:
                df[col] = to_numeric(df[col])
        if y and y != "count" and y in df.columns:
            df = df.dropna(subset=[y])

        if spec.resample and x:
            df = ChartService.bucket_dates(df, x, spec.resample)

        if spec.aggregation != "none":
            keys = [x]
            if spec.color and spec.color in df.columns and spec.color != x:
                keys.append(spec.color)

            if spec.ratio and y == spec.ratio["name"]:
                # A rate aggregates as sum(numerator)/sum(denominator). Averaging the
                # per-row ratios would weight a 2-unit row like a 2000-unit one.
                pair = [spec.ratio["num"], spec.ratio["den"]]
                df = df.groupby(keys, sort=False)[pair].sum().reset_index()
                df[y] = df[spec.ratio["num"]] / df[spec.ratio["den"]].replace(0, np.nan)
                logger.info(f"  ➗ Aggregated rate by {keys}")
            elif spec.aggregation == "count":
                df = df.groupby(keys, sort=False).size().reset_index(name="count")
                spec.y = y = "count"
                logger.info(f"  📊 Aggregated: count by {keys}")
            else:
                how = {
                    c: spec.aggregation for c in spec.value_columns if c in df.columns
                }
                # A secondary axis exists to carry a different *kind* of measure — a
                # percentage, a rate, an index. Summing one is meaningless: margin_pct
                # summed over 48 rows per region put 990 on the right-hand axis and read
                # as a margin collapse. It gets its own function, mean unless asked.
                if spec.y2 and spec.y2 in how:
                    how[spec.y2] = spec.y2_aggregation
                if how:
                    df = df.groupby(keys, sort=False).agg(how).reset_index()
                    logger.info(f"  📊 Aggregated: {how} by {keys}")

        # "Top 10 products by revenue" is a ranking, not a full plot
        if spec.top_n and y and y in df.columns:
            if len(df) > int(spec.top_n):
                spec.trimmed_from = len(df)
            df = df.nlargest(int(spec.top_n), y)
            logger.info(f"  🔝 Trimmed to top {spec.top_n} by {y}")

        return df

    @staticmethod
    def _auto_select_y_column(df: pd.DataFrame, chart_type: str) -> Optional[str]:
        """Auto-select Y column for charts that need one"""
        if chart_type == "histogram":
            return None  # Histogram doesn't need Y

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            selected = numeric_cols[0]
            logger.info(f"  🎯 Auto-selected Y column: {selected}")
            return selected

        return None

    @staticmethod
    def _calculate_summary_stats(
        df: pd.DataFrame, y_col: Optional[str], x_col: Optional[str] = None
    ) -> str:
        """
        Describe the figure to whoever has to write about it.

        The values are the ones plotted, after normalising, aggregating and trimming —
        which is why they belong in the receipt. Given only "Max/Min/Avg" the model wrote
        its prose from an earlier statistics call instead, and named a leader at 282.32
        above a bar reading 273.2. A small categorical chart therefore reports every
        point, so the text has the drawn numbers to quote.
        """
        if not y_col or df.empty or y_col not in df.columns:
            return "Chart created successfully."

        try:
            values = to_numeric(df[y_col]).dropna()
            if values.empty:
                return "Chart created successfully."

            summary = (
                f"Max: {values.max():.2f}, Min: {values.min():.2f}, "
                f"Avg: {values.mean():.2f}"
            )

            if x_col and x_col in df.columns and len(df) <= 12:
                points = ", ".join(
                    f"{label}={value:.2f}"
                    for label, value in zip(df[x_col].astype(str), to_numeric(df[y_col]))
                    if pd.notna(value)
                )
                if points:
                    summary = f"Plotted: {points}. {summary}"

            logger.info(f"  📈 Stats: {summary[:160]}")
            return summary
        except Exception as e:
            logger.debug(f"  ℹ️  Stats calculation skipped: {e}")
            return "Chart created successfully."

    @staticmethod
    def _value_format(df: pd.DataFrame, y_col: Optional[str]) -> str:
        """
        Pick a label format from the magnitude of the values.

        SI notation is right for revenue (1.04M) and actively misleading for a rate:
        it renders 0.0369 as "36.9m", meaning milli, next to charts labelled in
        millions. Fractions read as percentages instead.
        """
        if not y_col or y_col not in df.columns:
            return ".4g"

        try:
            peak = to_numeric(df[y_col]).abs().max()
        except Exception:
            return ".4g"

        if pd.isna(peak):
            return ".4g"
        if peak < 1:
            return ".2%"
        if peak >= 10_000:
            return ".3s"
        return ".4g"

    @staticmethod
    def _create_plotly_figure(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
        """Create the Plotly figure for a resolved spec."""
        x, y, color, title = spec.x, spec.y, spec.color, spec.title
        kind = spec.chart_type

        logger.info(f"  🎨 Creating {kind} chart: {title}")
        value_fmt = ChartService._value_format(df, y)

        if kind == "bar":
            fig = px.bar(df, x=x, y=y, title=title, color=color, text_auto=value_fmt)
            fig.update_yaxes(tickformat=value_fmt)

        elif kind == "line":
            fig = px.line(df, x=x, y=y, title=title, color=color, markers=True)

        elif kind == "pie":
            fig = px.pie(df.head(10), names=x, values=y, title=title, hole=0.3)

        elif kind == "scatter":
            fig = px.scatter(
                df,
                x=x,
                y=y,
                title=title,
                color=color,
                size=y if y else None,
                hover_data=df.columns.tolist(),
            )

        elif kind == "histogram":
            fig = px.histogram(df, x=x, title=title, color=color, marginal="box")

        elif kind == "box":
            fig = px.box(df, x=x, y=y, title=title, color=color, points="outliers")

        elif kind == "heatmap":
            # Correlation is the one statistic that is unreadable as text
            fig = px.imshow(
                df.select_dtypes(include=[np.number]).corr(),
                title=title,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
            )

        else:
            # Quietly drawing a bar chart for an unrecognised type hands back a figure
            # that answers a different question than the one asked.
            raise ValueError(
                f"Unsupported chart_type '{kind}'. Supported: "
                "bar, line, pie, scatter, histogram, box, heatmap"
            )

        # Side-by-side reads better than stacked when a series is broken out
        if color and kind == "bar":
            fig.update_layout(barmode="group")

        return ChartService._add_secondary_axis(fig, df, spec)

    @staticmethod
    def _add_secondary_axis(
        fig: go.Figure, df: pd.DataFrame, spec: ChartSpec
    ) -> go.Figure:
        """
        Plot a second measure against its own right-hand axis.

        Revenue and margin percentage belong on one chart but not one scale: sharing
        an axis flattens the smaller series into the baseline. A second axis is what
        makes the comparison honest.
        """
        if not spec.y2 or spec.y2 not in df.columns:
            return fig
        if spec.chart_type not in ("bar", "line", "scatter"):
            logger.warning(f"  ⚠️ Secondary axis ignored for {spec.chart_type}")
            return fig

        fig.add_trace(
            go.Scatter(
                x=df[spec.x],
                y=df[spec.y2],
                name=spec.y2,
                mode="lines+markers",
                yaxis="y2",
                line={"dash": "dot"},
            )
        )
        fig.update_layout(
            yaxis2={
                "title": spec.y2,
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "tickformat": ChartService._value_format(df, spec.y2),
            },
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        )
        logger.info(f"  ⇄ Secondary axis: {spec.y2}")
        return fig

    @staticmethod
    def _add_reference(
        fig: go.Figure, df: pd.DataFrame, spec: ChartSpec
    ) -> go.Figure:
        """
        Draw a baseline so a comparison answers "compared with what".

        Accepts "mean", "median" or a literal number.
        """
        if not spec.reference or not spec.y or spec.y not in df.columns:
            return fig

        series = to_numeric(df[spec.y])
        choice = str(spec.reference).strip().lower()

        if choice == "mean":
            value, label = series.mean(), "mean"
        elif choice == "median":
            value, label = series.median(), "median"
        else:
            try:
                value, label = float(spec.reference), "target"
            except ValueError:
                logger.warning(f"  ⚠️ Unusable reference '{spec.reference}'")
                return fig

        if pd.isna(value):
            return fig

        # The annotation has to agree with the axis it sits on. Both read from the same
        # format decision: a rate axis labelled "3.44%" beside "mean: 0.03" looks like a
        # reference to some other number entirely.
        text = f"{label}: " + ChartService._ANNOTATION_FORMATS.get(
            ChartService._value_format(df, spec.y), "{:,.2f}"
        ).format(value)

        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color="rgba(0,0,0,0.45)",
            annotation_text=text,
            annotation_position="top left",
        )
        logger.info(f"  📏 Reference line at {text}")
        return fig

    @staticmethod
    def _reference_split(df: pd.DataFrame, spec: ChartSpec, text: str) -> str:
        """
        Name which plotted points sit above the baseline and which below.

        The receipt already carried every bar and the average, and the model still wrote
        that two regions were above a line only one of them cleared — 423,699 read as
        above 428,737. It had the figures; comparing them was the step that failed. So the
        comparison is done here, where it is arithmetic rather than recollection.
        """
        if not spec.x or spec.x not in df.columns or not spec.y or spec.y not in df.columns:
            return ""

        try:
            threshold = float(text.split(":")[-1].strip().replace(",", "").rstrip("%"))
        except ValueError:
            return ""

        values = to_numeric(df[spec.y])
        labels = df[spec.x].astype(str)
        if values.isna().all() or len(df) > 30:
            return ""

        above = [l for l, v in zip(labels, values) if pd.notna(v) and v > threshold]
        below = [l for l, v in zip(labels, values) if pd.notna(v) and v <= threshold]
        return (
            f" Above it: {', '.join(above) or 'none'}."
            f" At or below: {', '.join(below) or 'none'}."
        )

    @staticmethod
    def _apply_layout_enhancements(fig: go.Figure, chart_type: str) -> go.Figure:
        """Apply layout enhancements for better UX"""

        # Apply default layout
        fig.update_layout(**ChartService.DEFAULT_LAYOUT)

        # Chart-specific enhancements
        if chart_type in ["bar", "line", "scatter"]:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(0,0,0,0.1)",
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(0,0,0,0.1)",
            )

        # Make it responsive
        fig.update_layout(
            autosize=True,
            showlegend=True,
        )

        return fig

    @staticmethod
    def generate_chart(df: pd.DataFrame, **request) -> Dict[str, Any]:
        """
        Generate a Plotly chart.

        Accepts the tool vocabulary (chart_type, x_column, y_column, y2_column,
        color_column, aggregation, top_n, resample, reference, title) and returns:

            {"success", "chart_json", "chart_html", "summary"}  or  {"success", "error"}
        """
        try:
            plot_df = df.copy().loc[:, ~df.columns.duplicated()]
            spec = ChartService.resolve(plot_df, **request)

            logger.info("=" * 60)
            logger.info(f"📊 CHART: {spec.chart_type} · {spec.title}")
            logger.info(f"  x={spec.x} y={spec.y} y2={spec.y2} color={spec.color}")
            logger.info(f"  agg={spec.aggregation} top_n={spec.top_n} resample={spec.resample}")
            logger.info(f"  rows={len(plot_df)}")
            logger.info("=" * 60)

            # A correlation heatmap spans every numeric column, so it has no x-axis
            if spec.chart_type == "heatmap":
                if plot_df.select_dtypes(include=[np.number]).shape[1] < 2:
                    return {
                        "success": False,
                        "error": "Heatmap needs at least two numeric columns",
                    }
            elif not spec.x:
                requested = request.get("x_column")
                error = f"Column '{requested}' not found. Available: {list(plot_df.columns)}"
                logger.error(f"  ❌ {error}")
                return {"success": False, "error": error}

            if spec.x:
                plot_df = ChartService._clean_and_sort_data(plot_df, spec.x)
                plot_df = ChartService._prepare_data(plot_df, spec)

            if plot_df.empty:
                logger.error("  ❌ No data remaining after processing")
                return {"success": False, "error": "No data remaining after processing"}

            if not spec.y and spec.chart_type not in ("histogram", "heatmap"):
                # Auto-selection is for a request that omits y. A y that was *named* and
                # did not resolve is a different situation: falling through to the first
                # numeric column — or to a row count — answers a question nobody asked,
                # and the chart looks entirely convincing.
                requested = request.get("y_column")
                if requested:
                    error = (
                        f"Column '{requested}' not found. "
                        f"Available: {list(plot_df.columns)}"
                    )
                    logger.error(f"  ❌ {error}")
                    return {"success": False, "error": error}
                spec.y = ChartService._auto_select_y_column(plot_df, spec.chart_type)

            fig = ChartService._create_plotly_figure(plot_df, spec)
            fig = ChartService._add_reference(fig, plot_df, spec)
            fig = ChartService._apply_layout_enhancements(fig, spec.chart_type)

            summary = ChartService._calculate_summary_stats(plot_df, spec.y, spec.x)

            # "Which region leads?" was answered in prose about four regions above a chart
            # of one, because top_n=1 trimmed the figure and nothing said so.
            if spec.trimmed_from:
                summary = (
                    f"{summary} Shows only the top {len(plot_df)} of "
                    f"{spec.trimmed_from} groups — say so, or plot them all."
                )

            # Where the baseline was actually drawn. A model that passes a figure of its
            # own then describes "the average" as that figure, while the bars average
            # something else; with both in the receipt the discrepancy is at least visible.
            drawn = (fig.layout.annotations or [{}])[0]
            if getattr(drawn, "text", None):
                summary = f"{summary} Reference line: {drawn.text}."
                summary += ChartService._reference_split(plot_df, spec, drawn.text)

            logger.info(f"  ✅ Chart ready ({len(plot_df)} rows plotted)")
            logger.info("=" * 60)

            return {
                "success": True,
                "chart_json": json.loads(fig.to_json()),
                "chart_html": fig.to_html(
                    include_plotlyjs="cdn", config=ChartService.CHART_CONFIG
                ),
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    MAX_DASHBOARD_CHARTS = 6

    @staticmethod
    def generate_multiple_charts(
        df: pd.DataFrame,
        chart_configs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate several charts from one dataset.

        One bad specification must not lose the others, so each is attempted
        independently and both outcomes are reported:

            {"success": bool, "charts": [...], "failures": [{"title", "error"}]}

        `success` is False only when nothing could be built.
        """
        requested = chart_configs[: ChartService.MAX_DASHBOARD_CHARTS]
        logger.info(f"📊 Dashboard: building {len(requested)} charts")

        charts, failures = [], []

        for i, config in enumerate(requested, 1):
            title = config.get("title") or f"Chart {i}"

            try:
                result = ChartService.generate_chart(df, **config)
            except TypeError as e:
                result = {"success": False, "error": f"Unsupported option: {e}"}

            if result.get("success"):
                charts.append(
                    {
                        "title": title,
                        "chart_json": result["chart_json"],
                        "chart_html": result["chart_html"],
                        "summary": result["summary"],
                    }
                )
            else:
                logger.warning(f"  ⚠️ '{title}' failed: {result.get('error')}")
                failures.append({"title": title, "error": result.get("error")})

        logger.info(f"  ✅ {len(charts)} built, {len(failures)} failed")

        return {
            "success": bool(charts),
            "charts": charts,
            "failures": failures,
        }
