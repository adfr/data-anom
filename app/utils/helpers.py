"""Helper utility functions."""

import re
from enum import Enum
from typing import Any, List, Optional

import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Enumeration of supported column types."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TEXT = "text"
    FREE_TEXT = "free_text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"


def detect_column_type(series: pd.Series, column_name: str = "") -> ColumnType:
    """
    Automatically detect the type of a pandas Series.

    Args:
        series: The pandas Series to analyze
        column_name: Optional column name for pattern matching

    Returns:
        ColumnType indicating the detected type
    """
    col_lower = column_name.lower()

    # Check for ID columns by name
    if any(pattern in col_lower for pattern in ["_id", "id_", "uuid", "guid"]):
        return ColumnType.ID

    # Check for email pattern
    if "email" in col_lower:
        return ColumnType.EMAIL

    # Check for phone pattern
    if any(pattern in col_lower for pattern in ["phone", "mobile", "tel"]):
        return ColumnType.PHONE

    # Check for name pattern
    if any(pattern in col_lower for pattern in ["name", "first_name", "last_name", "full_name"]):
        return ColumnType.NAME

    # Check for address pattern
    if any(pattern in col_lower for pattern in ["address", "street", "city", "zip", "postal"]):
        return ColumnType.ADDRESS

    # Check pandas dtype
    dtype = series.dtype

    # Boolean
    if dtype == bool or series.dropna().isin([True, False, 0, 1]).all():
        if series.nunique() <= 2:
            return ColumnType.BOOLEAN

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ColumnType.DATETIME

    # Numeric types
    if pd.api.types.is_numeric_dtype(dtype):
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # If very few unique values relative to total, likely categorical
        if series.nunique() <= 20 or (unique_ratio < 0.05 and series.nunique() < 50):
            return ColumnType.CATEGORICAL

        return ColumnType.CONTINUOUS

    # String types
    if pd.api.types.is_string_dtype(dtype) or dtype == object:
        # Sample non-null values for analysis
        sample = series.dropna().head(100)

        if len(sample) == 0:
            return ColumnType.TEXT

        # Check for email patterns in values
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if sample.astype(str).str.match(email_pattern).mean() > 0.8:
            return ColumnType.EMAIL

        # Calculate average word count
        avg_words = sample.astype(str).str.split().str.len().mean()

        # Calculate unique ratio
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # Free text: long strings with many words
        if avg_words > 10:
            return ColumnType.FREE_TEXT

        # Categorical: limited unique values
        if series.nunique() <= 50 or unique_ratio < 0.1:
            return ColumnType.CATEGORICAL

        # Short text with high cardinality
        if avg_words <= 5:
            return ColumnType.TEXT

        return ColumnType.FREE_TEXT

    return ColumnType.TEXT


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format a number with thousands separators.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if pd.isna(num):
        return "N/A"

    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.{decimals}f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.{decimals}f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.{decimals}f}K"
    else:
        return f"{num:,.{decimals}f}"


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string with ellipsis if needed
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def get_sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Get a sample of data for preview.

    Args:
        df: DataFrame to sample
        n: Number of rows to sample

    Returns:
        Sampled DataFrame
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def calculate_statistics(series: pd.Series) -> dict:
    """
    Calculate descriptive statistics for a series.

    Args:
        series: Pandas Series to analyze

    Returns:
        Dictionary of statistics
    """
    stats = {
        "count": len(series),
        "null_count": series.isna().sum(),
        "null_percentage": (series.isna().sum() / len(series) * 100) if len(series) > 0 else 0,
        "unique_count": series.nunique(),
        "unique_percentage": (series.nunique() / len(series) * 100) if len(series) > 0 else 0,
    }

    if pd.api.types.is_bool_dtype(series.dtype):
        # Boolean columns - just show value counts
        stats.update(
            {
                "true_count": series.sum(),
                "false_count": (series == False).sum(),
                "true_percentage": (series.sum() / series.count() * 100) if series.count() > 0 else 0,
            }
        )
    elif pd.api.types.is_numeric_dtype(series.dtype):
        stats.update(
            {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "q25": series.quantile(0.25),
                "q75": series.quantile(0.75),
            }
        )
    elif pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        non_null = series.dropna()
        if len(non_null) > 0:
            stats.update(
                {
                    "avg_length": non_null.astype(str).str.len().mean(),
                    "min_length": non_null.astype(str).str.len().min(),
                    "max_length": non_null.astype(str).str.len().max(),
                    "most_common": non_null.value_counts().head(5).to_dict(),
                }
            )

    return stats
