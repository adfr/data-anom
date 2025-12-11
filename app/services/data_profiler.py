"""Data profiling service for analyzing datasets."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.utils.helpers import ColumnType, calculate_statistics, detect_column_type


class DataProfiler:
    """
    Service for profiling datasets and detecting column characteristics.

    Analyzes data to understand distributions, patterns, and recommend
    appropriate synthetic data generation strategies.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data profiler.

        Args:
            df: DataFrame to profile
        """
        self.df = df
        self._profile: Optional[Dict[str, Any]] = None

    def profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.

        Returns:
            Dictionary containing dataset profile
        """
        if self._profile is not None:
            return self._profile

        logger.info(f"Profiling dataset with {len(self.df)} rows and {len(self.df.columns)} columns")

        profile = {
            "overview": self._get_overview(),
            "columns": {},
            "correlations": None,
            "recommendations": [],
        }

        # Profile each column
        for col in self.df.columns:
            profile["columns"][col] = self._profile_column(col)

        # Calculate correlations for numeric columns
        profile["correlations"] = self._calculate_correlations()

        # Generate recommendations
        profile["recommendations"] = self._generate_recommendations(profile)

        self._profile = profile
        return profile

    def _get_overview(self) -> Dict[str, Any]:
        """Get dataset overview statistics."""
        return {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": self.df.duplicated().sum(),
            "complete_rows": len(self.df.dropna()),
            "total_missing": self.df.isna().sum().sum(),
        }

    def _profile_column(self, column: str) -> Dict[str, Any]:
        """
        Profile a single column.

        Args:
            column: Column name

        Returns:
            Column profile dictionary
        """
        series = self.df[column]
        detected_type = detect_column_type(series, column)
        stats = calculate_statistics(series)

        profile = {
            "name": column,
            "dtype": str(series.dtype),
            "detected_type": detected_type.value,
            "statistics": stats,
            "sample_values": series.dropna().head(5).tolist(),
        }

        # Add type-specific analysis
        if detected_type == ColumnType.CONTINUOUS:
            profile["distribution"] = self._analyze_distribution(series)

        elif detected_type == ColumnType.CATEGORICAL:
            profile["categories"] = self._analyze_categories(series)

        elif detected_type in [ColumnType.TEXT, ColumnType.FREE_TEXT]:
            profile["text_analysis"] = self._analyze_text(series)

        elif detected_type == ColumnType.DATETIME:
            profile["temporal_analysis"] = self._analyze_temporal(series)

        return profile

    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution of continuous variable."""
        clean = series.dropna()

        if len(clean) == 0:
            return {"type": "unknown"}

        # Calculate distribution metrics
        skewness = clean.skew()
        kurtosis = clean.kurtosis()

        # Determine distribution type
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            dist_type = "normal"
        elif skewness > 1:
            dist_type = "right_skewed"
        elif skewness < -1:
            dist_type = "left_skewed"
        elif clean.min() >= 0 and skewness > 0.5:
            dist_type = "exponential"
        else:
            dist_type = "other"

        # Calculate percentiles
        percentiles = clean.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()

        return {
            "type": dist_type,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "percentiles": percentiles,
            "is_integer": (clean % 1 == 0).all(),
            "has_negative": (clean < 0).any(),
        }

    def _analyze_categories(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical variable."""
        value_counts = series.value_counts()

        return {
            "n_categories": series.nunique(),
            "value_counts": value_counts.head(20).to_dict(),
            "frequencies": (value_counts / len(series)).head(20).to_dict(),
            "rare_categories": len(value_counts[value_counts < len(series) * 0.01]),
        }

    def _analyze_text(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text column."""
        clean = series.dropna().astype(str)

        if len(clean) == 0:
            return {}

        lengths = clean.str.len()
        word_counts = clean.str.split().str.len()

        return {
            "avg_length": lengths.mean(),
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "avg_words": word_counts.mean(),
            "contains_digits": clean.str.contains(r"\d").mean(),
            "contains_special": clean.str.contains(r"[^a-zA-Z0-9\s]").mean(),
        }

    def _analyze_temporal(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze temporal column."""
        clean = pd.to_datetime(series.dropna(), errors="coerce").dropna()

        if len(clean) == 0:
            return {}

        return {
            "min_date": clean.min().isoformat(),
            "max_date": clean.max().isoformat(),
            "date_range_days": (clean.max() - clean.min()).days,
            "has_time": (clean.dt.hour != 0).any() or (clean.dt.minute != 0).any(),
        }

    def _calculate_correlations(self) -> Optional[Dict[str, Any]]:
        """Calculate correlations between numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        corr_matrix = self.df[numeric_cols].corr()

        # Find high correlations
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.5:
                    high_correlations.append(
                        {"column1": col1, "column2": col2, "correlation": round(corr, 3)}
                    )

        return {
            "matrix": corr_matrix.to_dict(),
            "high_correlations": sorted(
                high_correlations, key=lambda x: abs(x["correlation"]), reverse=True
            ),
        }

    def _generate_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations for synthetic data generation."""
        recommendations = []

        for col_name, col_profile in profile["columns"].items():
            detected_type = col_profile["detected_type"]
            stats = col_profile["statistics"]

            # High null percentage warning
            if stats.get("null_percentage", 0) > 20:
                recommendations.append(
                    {
                        "column": col_name,
                        "type": "warning",
                        "message": f"High null percentage ({stats['null_percentage']:.1f}%). Consider handling missing values.",
                    }
                )

            # Recommendation based on type
            if detected_type == ColumnType.ID.value:
                recommendations.append(
                    {
                        "column": col_name,
                        "type": "info",
                        "message": "Detected as ID column. Will generate unique sequential or UUID values.",
                    }
                )

            elif detected_type == ColumnType.EMAIL.value:
                recommendations.append(
                    {
                        "column": col_name,
                        "type": "info",
                        "message": "Detected as email. Will generate realistic fake email addresses.",
                    }
                )

            elif detected_type == ColumnType.FREE_TEXT.value:
                recommendations.append(
                    {
                        "column": col_name,
                        "type": "info",
                        "message": "Detected as free text. Will use text generation to create similar content.",
                    }
                )

        return recommendations

    def get_column_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get recommended configuration for each column.

        Returns:
            Dictionary mapping column names to their recommended config
        """
        if self._profile is None:
            self.profile()

        config = {}
        for col_name, col_profile in self._profile["columns"].items():
            detected_type = col_profile["detected_type"]

            config[col_name] = {
                "name": col_name,
                "detected_type": detected_type,
                "generation_method": self._recommend_generation_method(detected_type, col_profile),
                "preserve_statistics": True,
                "include": True,
            }

        return config

    def _recommend_generation_method(
        self, detected_type: str, col_profile: Dict[str, Any]
    ) -> str:
        """Recommend generation method based on column type."""
        method_map = {
            ColumnType.CONTINUOUS.value: "gaussian_copula",
            ColumnType.CATEGORICAL.value: "frequency_sampling",
            ColumnType.TEXT.value: "faker_text",
            ColumnType.FREE_TEXT.value: "markov_chain",
            ColumnType.DATETIME.value: "datetime_range",
            ColumnType.BOOLEAN.value: "bernoulli",
            ColumnType.ID.value: "uuid",
            ColumnType.EMAIL.value: "faker_email",
            ColumnType.PHONE.value: "faker_phone",
            ColumnType.ADDRESS.value: "faker_address",
            ColumnType.NAME.value: "faker_name",
        }
        return method_map.get(detected_type, "sample")

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics as a DataFrame.

        Returns:
            DataFrame with column statistics
        """
        if self._profile is None:
            self.profile()

        rows = []
        for col_name, col_profile in self._profile["columns"].items():
            stats = col_profile["statistics"]
            row = {
                "Column": col_name,
                "Type": col_profile["detected_type"],
                "Non-Null": stats["count"] - stats["null_count"],
                "Null %": f"{stats['null_percentage']:.1f}%",
                "Unique": stats["unique_count"],
            }

            if "mean" in stats:
                row["Mean"] = f"{stats['mean']:.2f}"
                row["Std"] = f"{stats['std']:.2f}"

            rows.append(row)

        return pd.DataFrame(rows)
