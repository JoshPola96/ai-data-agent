# app/services/chart.py

"""
Plotly Chart Service - Production Ready with Enhanced Visualizations
Supports multiple charts, responsive sizing, and rich metadata
"""

import logging
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
from difflib import get_close_matches
import json

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Parsing dates.*dayfirst.*")


class ChartService:
    """Enhanced chart generation service with Plotly"""

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
    def _prepare_data(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: Optional[str],
        aggregation: str,
    ) -> pd.DataFrame:
        """Prepare data with cleaning and aggregation"""

        # Categorical charts need string x-axis
        if chart_type in ["bar", "pie"]:
            df[x_col] = df[x_col].astype(str).fillna("Unknown")

        # Numeric columns
        if y_col and y_col != "count":
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            df = df.dropna(subset=[y_col])

        # Aggregation
        if aggregation != "none":
            if aggregation == "count":
                df = df.groupby(x_col, sort=False).size().reset_index(name="count")
                y_col = "count"
                logger.info(f"  📊 Aggregated: count by {x_col}")
            elif y_col:
                df = df.groupby(x_col, sort=False)[y_col].agg(aggregation).reset_index()
                logger.info(f"  📊 Aggregated: {aggregation}({y_col}) by {x_col}")

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
    def _calculate_summary_stats(df: pd.DataFrame, y_col: Optional[str]) -> str:
        """Calculate summary statistics for the chart"""
        if not y_col or df.empty:
            return "Chart created successfully."

        try:
            max_val = df[y_col].max()
            min_val = df[y_col].min()
            avg_val = df[y_col].mean()

            summary = f"Max: {max_val:.2f}, Min: {min_val:.2f}, Avg: {avg_val:.2f}"
            logger.info(f"  📈 Stats: {summary}")
            return summary
        except Exception as e:
            logger.debug(f"  ℹ️  Stats calculation skipped: {e}")
            return "Chart created successfully."

    @staticmethod
    def _create_plotly_figure(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: Optional[str],
        title: str,
        color_column: Optional[str] = None,
    ) -> go.Figure:
        """Create Plotly figure based on chart type"""

        logger.info(f"  🎨 Creating {chart_type} chart: {title}")

        if chart_type == "bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color=color_column,
                text_auto=True,  # Show values on bars
            )

        elif chart_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color=color_column,
                markers=True,  # Show data points
            )

        elif chart_type == "pie":
            # Limit to top 10 for readability
            plot_df = df.head(10)
            fig = px.pie(
                plot_df,
                names=x_col,
                values=y_col,
                title=title,
                hole=0.3,  # Donut chart for better aesthetics
            )

        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color=color_column,
                size=y_col if y_col else None,  # Size by value
                hover_data=df.columns.tolist(),  # Show all data on hover
            )

        elif chart_type == "histogram":
            fig = px.histogram(
                df,
                x=x_col,
                title=title,
                color=color_column,
                marginal="box",  # Add box plot on top
            )

        else:
            # Default to bar chart
            fig = px.bar(df, x=x_col, y=y_col, title=title)

        return fig

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
    def generate_chart(
        df: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_column: Optional[str] = None,
        aggregation: str = "none",
        title: str = "Chart",
        color_column: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a Plotly chart with enhanced features.

        Returns:
            {
                "success": bool,
                "chart_json": dict,  # Plotly JSON for programmatic use
                "chart_html": str,   # Standalone HTML
                "summary": str,      # Summary statistics
                "error": str,        # Error message if failed
            }
        """
        try:
            logger.info("=" * 60)
            logger.info("📊 CHART GENERATION REQUEST")
            logger.info(f"  Type: {chart_type}")
            logger.info(f"  X: {x_column}")
            logger.info(f"  Y: {y_column}")
            logger.info(f"  Aggregation: {aggregation}")
            logger.info(f"  Title: {title}")
            logger.info("=" * 60)

            # Remove duplicate columns
            plot_df = df.copy().loc[:, ~df.columns.duplicated()]
            logger.info(f"  📋 Input shape: {plot_df.shape}")

            # Match columns
            x_col = ChartService._fuzzy_col_match(plot_df, x_column)
            y_col = (
                ChartService._fuzzy_col_match(plot_df, y_column) if y_column else None
            )

            if not x_col:
                error_msg = (
                    f"Column '{x_column}' not found. Available: {list(plot_df.columns)}"
                )
                logger.error(f"  ❌ {error_msg}")
                return {"success": False, "error": error_msg}

            # Clean and sort
            plot_df = ChartService._clean_and_sort_data(plot_df, x_col)

            # Prepare data
            plot_df = ChartService._prepare_data(
                plot_df, chart_type, x_col, y_col, aggregation
            )

            if plot_df.empty:
                error_msg = "No data remaining after processing"
                logger.error(f"  ❌ {error_msg}")
                return {"success": False, "error": error_msg}

            logger.info(f"  ✅ Processed shape: {plot_df.shape}")

            # Auto-select Y if needed
            if not y_col and chart_type != "histogram":
                y_col = ChartService._auto_select_y_column(plot_df, chart_type)
                if aggregation == "count":
                    y_col = "count"

            # Create figure
            fig = ChartService._create_plotly_figure(
                plot_df, chart_type, x_col, y_col, title, color_column
            )

            # Apply enhancements
            fig = ChartService._apply_layout_enhancements(fig, chart_type)

            # Calculate summary
            summary = ChartService._calculate_summary_stats(plot_df, y_col)

            # Serialize to JSON
            chart_json_str = fig.to_json()
            chart_json = json.loads(chart_json_str)

            # Generate standalone HTML
            chart_html = fig.to_html(
                include_plotlyjs="cdn",
                config=ChartService.CHART_CONFIG,
            )

            logger.info("  ✅ Chart generated successfully")
            logger.info("=" * 60)

            return {
                "success": True,
                "chart_json": chart_json,
                "chart_html": chart_html,
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
