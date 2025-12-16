"""Reusable UI components for Streamlit application."""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.helpers import ColumnType


def display_data_preview(
    df: pd.DataFrame,
    title: str = "Data Preview",
    max_rows: int = 10,
) -> None:
    """
    Display a preview of the DataFrame.

    Args:
        df: DataFrame to display
        title: Section title
        max_rows: Maximum rows to show
    """
    st.subheader(title)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.2f} MB")

    st.dataframe(df.head(max_rows), use_container_width=True)


def display_profile_summary(profile: Dict[str, Any]) -> None:
    """
    Display data profile summary.

    Args:
        profile: Profile dictionary from DataProfiler
    """
    st.subheader("Dataset Overview")

    overview = profile.get("overview", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{overview.get('n_rows', 0):,}")
    with col2:
        st.metric("Total Columns", overview.get("n_columns", 0))
    with col3:
        st.metric("Complete Rows", f"{overview.get('complete_rows', 0):,}")
    with col4:
        st.metric("Total Missing", f"{overview.get('total_missing', 0):,}")

    # Column type distribution
    st.subheader("Column Types")

    type_counts = {}
    for col_profile in profile.get("columns", {}).values():
        col_type = col_profile.get("detected_type", "unknown")
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

    if type_counts:
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Type Distribution",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    recommendations = profile.get("recommendations", [])
    if recommendations:
        st.subheader("Recommendations")
        for rec in recommendations[:5]:
            icon = "!" if rec.get("type") == "warning" else "i"
            st.info(f"**{rec.get('column')}**: {rec.get('message')}")


def display_column_config(
    columns: Dict[str, Dict[str, Any]],
    editable: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Display and optionally edit column configuration.

    Args:
        columns: Column configuration dictionary
        editable: Whether to allow editing

    Returns:
        Updated column configuration
    """
    st.subheader("Column Configuration")

    updated_config = {}

    # Create a table-like view
    for col_name, col_config in columns.items():
        with st.expander(f"{col_name} ({col_config.get('detected_type', 'unknown')})", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                include = st.checkbox(
                    "Include in output",
                    value=col_config.get("include", True),
                    key=f"include_{col_name}",
                    disabled=not editable,
                )

            with col2:
                generation_methods = [
                    "auto",
                    "gaussian_copula",
                    "kde",
                    "frequency_sampling",
                    "faker_text",
                    "faker_email",
                    "faker_phone",
                    "faker_name",
                    "faker_address",
                    "markov_chain",
                    "datetime_range",
                    "bernoulli",
                    "uuid",
                    "sequential_id",
                    "sample",
                ]

                current_method = col_config.get("generation_method", "auto")
                method_index = (
                    generation_methods.index(current_method)
                    if current_method in generation_methods
                    else 0
                )

                method = st.selectbox(
                    "Generation Method",
                    generation_methods,
                    index=method_index,
                    key=f"method_{col_name}",
                    disabled=not editable,
                )

            preserve_stats = st.checkbox(
                "Preserve statistics",
                value=col_config.get("preserve_statistics", True),
                key=f"preserve_{col_name}",
                disabled=not editable,
            )

            updated_config[col_name] = {
                "name": col_name,
                "detected_type": col_config.get("detected_type"),
                "generation_method": method,
                "preserve_statistics": preserve_stats,
                "include": include,
            }

    return updated_config


def display_column_statistics(col_profile: Dict[str, Any]) -> None:
    """
    Display statistics for a single column.

    Args:
        col_profile: Column profile dictionary
    """
    stats = col_profile.get("statistics", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Non-Null", f"{stats.get('count', 0) - stats.get('null_count', 0):,}")

    with col2:
        st.metric("Null %", f"{stats.get('null_percentage', 0):.1f}%")

    with col3:
        st.metric("Unique", f"{stats.get('unique_count', 0):,}")

    # Numeric statistics
    if "mean" in stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{stats.get('mean', 0):.2f}")
        with col2:
            st.metric("Std", f"{stats.get('std', 0):.2f}")
        with col3:
            st.metric("Min", f"{stats.get('min', 0):.2f}")
        with col4:
            st.metric("Max", f"{stats.get('max', 0):.2f}")

    # Text statistics
    if "avg_length" in stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Length", f"{stats.get('avg_length', 0):.1f}")
        with col2:
            st.metric("Min Length", stats.get("min_length", 0))
        with col3:
            st.metric("Max Length", stats.get("max_length", 0))

    # Top categories
    if "most_common" in stats:
        st.write("**Most Common Values:**")
        for value, count in list(stats.get("most_common", {}).items())[:5]:
            st.write(f"- {value}: {count:,}")


def display_generation_progress(current: int, total: int, status: str = "Generating...") -> None:
    """
    Display progress during synthetic data generation.

    Args:
        current: Current progress
        total: Total items
        status: Status message
    """
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.write(f"{status} ({current}/{total})")


def display_quality_report(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Display comparison between original and synthetic data.

    Args:
        original_df: Original DataFrame
        synthetic_df: Synthetic DataFrame
    """
    st.subheader("Quality Report")

    # Basic comparison
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Data**")
        st.metric("Rows", f"{len(original_df):,}")
        st.metric("Columns", len(original_df.columns))

    with col2:
        st.write("**Synthetic Data**")
        st.metric("Rows", f"{len(synthetic_df):,}")
        st.metric("Columns", len(synthetic_df.columns))

    # Distribution comparison for numeric columns
    numeric_cols = original_df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        st.subheader("Distribution Comparison")

        selected_col = st.selectbox("Select column to compare", numeric_cols)

        if selected_col:
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=original_df[selected_col].dropna(),
                    name="Original",
                    opacity=0.7,
                    nbinsx=30,
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=synthetic_df[selected_col].dropna(),
                    name="Synthetic",
                    opacity=0.7,
                    nbinsx=30,
                )
            )

            fig.update_layout(
                title=f"Distribution Comparison: {selected_col}",
                barmode="overlay",
                xaxis_title=selected_col,
                yaxis_title="Count",
            )

            st.plotly_chart(fig, use_container_width=True)

    # Statistics comparison
    st.subheader("Statistical Comparison")

    comparison_data = []
    for col in original_df.columns:
        if col not in synthetic_df.columns:
            continue

        row = {"Column": col}

        if pd.api.types.is_numeric_dtype(original_df[col]):
            row["Orig Mean"] = f"{original_df[col].mean():.2f}"
            row["Synth Mean"] = f"{synthetic_df[col].mean():.2f}"
            row["Orig Std"] = f"{original_df[col].std():.2f}"
            row["Synth Std"] = f"{synthetic_df[col].std():.2f}"
        else:
            row["Orig Unique"] = original_df[col].nunique()
            row["Synth Unique"] = synthetic_df[col].nunique()

        comparison_data.append(row)

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


def display_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """
    Display correlation heatmap for numeric columns.

    Args:
        df: DataFrame
        title: Plot title
    """
    import numpy as np

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis")
        return

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=numeric_cols,
        y=numeric_cols,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def display_correlation_comparison(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Display side-by-side correlation comparison between original and synthetic data.

    Shows:
    - Side-by-side correlation heatmaps
    - Correlation preservation metrics
    - Detailed pair-wise correlation comparison

    Args:
        original_df: Original DataFrame
        synthetic_df: Synthetic DataFrame
    """
    import numpy as np
    from plotly.subplots import make_subplots

    st.subheader("Correlation Preservation Analysis")

    # Get numeric columns common to both DataFrames
    orig_numeric = set(original_df.select_dtypes(include=[np.number]).columns)
    synth_numeric = set(synthetic_df.select_dtypes(include=[np.number]).columns)
    numeric_cols = sorted(list(orig_numeric & synth_numeric))

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis (need at least 2)")
        return

    # Calculate correlation matrices
    orig_corr = original_df[numeric_cols].corr()
    synth_corr = synthetic_df[numeric_cols].corr()

    # Calculate correlation difference
    corr_diff = orig_corr - synth_corr

    # --- Metrics Section ---
    st.markdown("#### Preservation Metrics")

    # Calculate key metrics
    # Mean Absolute Error between correlation matrices
    mask = np.triu(np.ones_like(orig_corr, dtype=bool), k=1)
    orig_upper = orig_corr.values[mask]
    synth_upper = synth_corr.values[mask]

    mae = np.mean(np.abs(orig_upper - synth_upper))
    rmse = np.sqrt(np.mean((orig_upper - synth_upper) ** 2))
    max_diff = np.max(np.abs(orig_upper - synth_upper))

    # Correlation of correlations (how well are relationships preserved)
    corr_of_corr = np.corrcoef(orig_upper, synth_upper)[0, 1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Correlation of Correlations",
            f"{corr_of_corr:.3f}",
            help="How well the synthetic data preserves the relative strength of correlations (1.0 = perfect)",
        )

    with col2:
        st.metric(
            "Mean Absolute Error",
            f"{mae:.3f}",
            help="Average absolute difference between original and synthetic correlations (lower is better)",
        )

    with col3:
        st.metric(
            "RMSE",
            f"{rmse:.3f}",
            help="Root Mean Square Error of correlation differences (lower is better)",
        )

    with col4:
        st.metric(
            "Max Difference",
            f"{max_diff:.3f}",
            help="Largest absolute difference between any pair of correlations",
        )

    # Quality indicator
    if corr_of_corr > 0.9 and mae < 0.1:
        st.success("✓ Excellent correlation preservation")
    elif corr_of_corr > 0.7 and mae < 0.2:
        st.info("○ Good correlation preservation")
    elif corr_of_corr > 0.5:
        st.warning("△ Moderate correlation preservation - consider using SDV Gaussian Copula")
    else:
        st.error("✗ Poor correlation preservation - recommend using SDV generator")

    # --- Side-by-side Heatmaps ---
    st.markdown("#### Correlation Matrices Comparison")

    # Create subplots for side-by-side heatmaps
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Original Data", "Synthetic Data", "Difference (Orig - Synth)"),
        horizontal_spacing=0.08,
    )

    # Original correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=orig_corr.values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(orig_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Synthetic correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=synth_corr.values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(synth_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Difference heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_diff.values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_diff.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Diff", x=1.02),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        height=450,
        title_text="",
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Detailed Pair-wise Comparison ---
    st.markdown("#### Pair-wise Correlation Details")

    # Build comparison table
    pair_data = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            orig_val = orig_corr.loc[col1, col2]
            synth_val = synth_corr.loc[col1, col2]
            diff = orig_val - synth_val

            pair_data.append(
                {
                    "Column 1": col1,
                    "Column 2": col2,
                    "Original Corr": round(orig_val, 3),
                    "Synthetic Corr": round(synth_val, 3),
                    "Difference": round(diff, 3),
                    "Abs Difference": round(abs(diff), 3),
                    "Status": "✓" if abs(diff) < 0.1 else ("△" if abs(diff) < 0.2 else "✗"),
                }
            )

    if pair_data:
        pair_df = pd.DataFrame(pair_data)
        pair_df = pair_df.sort_values("Abs Difference", ascending=False)

        st.dataframe(
            pair_df,
            use_container_width=True,
            hide_index=True,
        )

    # --- Scatter plot of correlations ---
    st.markdown("#### Correlation Scatter Plot")
    st.caption("Each point represents a pair of columns. Perfect preservation = points on diagonal.")

    scatter_df = pd.DataFrame(
        {
            "Original Correlation": orig_upper,
            "Synthetic Correlation": synth_upper,
        }
    )

    fig_scatter = px.scatter(
        scatter_df,
        x="Original Correlation",
        y="Synthetic Correlation",
        title="Original vs Synthetic Correlations",
    )

    # Add diagonal reference line
    fig_scatter.add_shape(
        type="line",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
        line=dict(color="red", dash="dash"),
    )

    fig_scatter.update_layout(
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1]),
        height=400,
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
