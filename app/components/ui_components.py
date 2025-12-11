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
