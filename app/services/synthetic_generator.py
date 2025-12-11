"""Synthetic data generation service."""

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.utils.helpers import ColumnType


class SyntheticDataGenerator:
    """
    Generate synthetic data that preserves statistical properties of original data.

    Supports multiple generation strategies for different data types:
    - Continuous: Gaussian copula, kernel density estimation
    - Categorical: Frequency-based sampling
    - Text: Faker, Markov chains, sampling
    - Free text: Markov chains, template-based generation
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        column_config: Optional[Dict[str, Dict[str, Any]]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            source_df: Source DataFrame to learn from
            column_config: Configuration for each column
            random_seed: Random seed for reproducibility
        """
        self.source_df = source_df
        self.column_config = column_config or {}
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self._generators: Dict[str, Callable] = {}
        self._fitted = False

    def fit(self) -> "SyntheticDataGenerator":
        """
        Fit the generator to the source data.

        Learns distributions and patterns from the source data.

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting generator on {len(self.source_df)} rows")

        for col in self.source_df.columns:
            config = self.column_config.get(col, {})

            if not config.get("include", True):
                continue

            method = config.get("generation_method", "auto")
            detected_type = config.get("detected_type", "text")

            self._generators[col] = self._create_generator(col, method, detected_type)

        self._fitted = True
        logger.info(f"Fitted {len(self._generators)} column generators")
        return self

    def generate(self, n_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            n_rows: Number of rows to generate

        Returns:
            DataFrame with synthetic data
        """
        if not self._fitted:
            self.fit()

        logger.info(f"Generating {n_rows} synthetic rows")

        synthetic_data = {}

        for col, generator in self._generators.items():
            try:
                synthetic_data[col] = generator(n_rows)
            except Exception as e:
                logger.error(f"Error generating column {col}: {e}")
                # Fallback to sampling
                synthetic_data[col] = self._sample_with_replacement(col, n_rows)

        # Preserve correlations if configured
        df = pd.DataFrame(synthetic_data)

        # Ensure column order matches source
        cols_in_order = [c for c in self.source_df.columns if c in df.columns]
        df = df[cols_in_order]

        return df

    def _create_generator(
        self, column: str, method: str, detected_type: str
    ) -> Callable[[int], np.ndarray]:
        """
        Create a generator function for a column.

        Args:
            column: Column name
            method: Generation method
            detected_type: Detected column type

        Returns:
            Generator function that takes n_rows and returns values
        """
        series = self.source_df[column]

        # Auto-detect method based on type
        if method == "auto":
            method = self._auto_select_method(detected_type)

        # Map methods to generator functions
        generators = {
            "gaussian_copula": lambda n: self._generate_gaussian(series, n),
            "kde": lambda n: self._generate_kde(series, n),
            "frequency_sampling": lambda n: self._generate_categorical(series, n),
            "faker_text": lambda n: self._generate_faker_text(series, n),
            "faker_email": lambda n: self._generate_faker_email(n),
            "faker_phone": lambda n: self._generate_faker_phone(n),
            "faker_name": lambda n: self._generate_faker_name(n),
            "faker_address": lambda n: self._generate_faker_address(n),
            "markov_chain": lambda n: self._generate_markov(series, n),
            "datetime_range": lambda n: self._generate_datetime(series, n),
            "bernoulli": lambda n: self._generate_bernoulli(series, n),
            "uuid": lambda n: self._generate_uuid(n),
            "sequential_id": lambda n: self._generate_sequential_id(series, n),
            "sample": lambda n: self._sample_with_replacement(column, n),
        }

        return generators.get(method, lambda n: self._sample_with_replacement(column, n))

    def _auto_select_method(self, detected_type: str) -> str:
        """Auto-select generation method based on type."""
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

    def _generate_gaussian(self, series: pd.Series, n: int) -> np.ndarray:
        """Generate continuous values using Gaussian distribution."""
        clean = series.dropna()

        if len(clean) == 0:
            return np.full(n, np.nan)

        mean = clean.mean()
        std = clean.std()

        if std == 0 or np.isnan(std):
            return np.full(n, mean)

        # Generate from normal distribution
        values = np.random.normal(mean, std, n)

        # Clip to observed range (with some margin)
        min_val = clean.min()
        max_val = clean.max()
        margin = (max_val - min_val) * 0.1

        values = np.clip(values, min_val - margin, max_val + margin)

        # Preserve integer type if original was integer
        if (clean % 1 == 0).all():
            values = np.round(values).astype(int)

        # Add null values at original rate
        null_rate = series.isna().mean()
        if null_rate > 0:
            null_mask = np.random.random(n) < null_rate
            values = values.astype(float)
            values[null_mask] = np.nan

        return values

    def _generate_kde(self, series: pd.Series, n: int) -> np.ndarray:
        """Generate continuous values using Kernel Density Estimation."""
        from scipy import stats

        clean = series.dropna()

        if len(clean) < 10:
            return self._generate_gaussian(series, n)

        try:
            kde = stats.gaussian_kde(clean)
            values = kde.resample(n).flatten()

            # Clip to reasonable range
            min_val = clean.min()
            max_val = clean.max()
            values = np.clip(values, min_val, max_val)

            return values
        except Exception:
            return self._generate_gaussian(series, n)

    def _generate_categorical(self, series: pd.Series, n: int) -> np.ndarray:
        """Generate categorical values based on frequency distribution."""
        value_counts = series.value_counts(dropna=False)
        categories = value_counts.index.tolist()
        probabilities = (value_counts / value_counts.sum()).tolist()

        return np.random.choice(categories, size=n, p=probabilities)

    def _generate_faker_text(self, series: pd.Series, n: int) -> List[str]:
        """Generate short text using Faker."""
        from faker import Faker

        fake = Faker()

        # Analyze original text patterns
        clean = series.dropna().astype(str)
        avg_words = clean.str.split().str.len().mean() if len(clean) > 0 else 3

        if avg_words <= 2:
            return [fake.word() for _ in range(n)]
        elif avg_words <= 5:
            return [fake.sentence(nb_words=int(avg_words)) for _ in range(n)]
        else:
            return [fake.sentence(nb_words=int(avg_words)) for _ in range(n)]

    def _generate_faker_email(self, n: int) -> List[str]:
        """Generate fake email addresses."""
        from faker import Faker

        fake = Faker()
        return [fake.email() for _ in range(n)]

    def _generate_faker_phone(self, n: int) -> List[str]:
        """Generate fake phone numbers."""
        from faker import Faker

        fake = Faker()
        return [fake.phone_number() for _ in range(n)]

    def _generate_faker_name(self, n: int) -> List[str]:
        """Generate fake names."""
        from faker import Faker

        fake = Faker()
        return [fake.name() for _ in range(n)]

    def _generate_faker_address(self, n: int) -> List[str]:
        """Generate fake addresses."""
        from faker import Faker

        fake = Faker()
        return [fake.address().replace("\n", ", ") for _ in range(n)]

    def _generate_markov(self, series: pd.Series, n: int) -> List[str]:
        """Generate text using simple Markov chain."""
        clean = series.dropna().astype(str).tolist()

        if len(clean) < 10:
            return self._sample_with_replacement_list(clean, n)

        # Build simple word-level Markov chain
        transitions: Dict[str, List[str]] = {}

        for text in clean:
            words = text.split()
            if len(words) < 2:
                continue

            for i in range(len(words) - 1):
                current = words[i].lower()
                next_word = words[i + 1]

                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_word)

        if not transitions:
            return self._sample_with_replacement_list(clean, n)

        # Generate new texts
        avg_len = int(pd.Series([len(t.split()) for t in clean]).mean())
        results = []

        for _ in range(n):
            # Start with random word
            current = np.random.choice(list(transitions.keys()))
            words = [current.capitalize()]

            for _ in range(avg_len - 1):
                current_lower = current.lower()
                if current_lower in transitions:
                    next_word = np.random.choice(transitions[current_lower])
                    words.append(next_word)
                    current = next_word
                else:
                    # Random restart
                    current = np.random.choice(list(transitions.keys()))
                    words.append(current)

            results.append(" ".join(words))

        return results

    def _generate_datetime(self, series: pd.Series, n: int) -> pd.Series:
        """Generate datetime values within observed range."""
        clean = pd.to_datetime(series.dropna(), errors="coerce").dropna()

        if len(clean) == 0:
            # Default to last year
            end = datetime.now()
            start = end - timedelta(days=365)
        else:
            start = clean.min()
            end = clean.max()

        # Generate random timestamps
        delta = end - start
        random_seconds = np.random.randint(0, int(delta.total_seconds()) + 1, n)
        dates = [start + timedelta(seconds=int(s)) for s in random_seconds]

        return pd.Series(dates)

    def _generate_bernoulli(self, series: pd.Series, n: int) -> np.ndarray:
        """Generate boolean values based on observed probability."""
        clean = series.dropna()

        if len(clean) == 0:
            p = 0.5
        else:
            # Calculate probability of True/1
            p = clean.astype(bool).mean()

        return np.random.random(n) < p

    def _generate_uuid(self, n: int) -> List[str]:
        """Generate unique UUIDs."""
        return [str(uuid.uuid4()) for _ in range(n)]

    def _generate_sequential_id(self, series: pd.Series, n: int) -> List[str]:
        """Generate sequential IDs matching original pattern."""
        sample = series.dropna().astype(str).iloc[0] if len(series.dropna()) > 0 else "ID_00001"

        # Try to detect pattern
        match = re.match(r"([A-Za-z_]*)(\d+)", sample)

        if match:
            prefix = match.group(1)
            num_digits = len(match.group(2))
            return [f"{prefix}{i:0{num_digits}d}" for i in range(1, n + 1)]

        return [f"ID_{i:05d}" for i in range(1, n + 1)]

    def _sample_with_replacement(self, column: str, n: int) -> np.ndarray:
        """Sample from original data with replacement."""
        return self.source_df[column].sample(n=n, replace=True).values

    def _sample_with_replacement_list(self, values: List, n: int) -> List:
        """Sample from list with replacement."""
        return list(np.random.choice(values, size=n, replace=True))


class SDVSyntheticGenerator:
    """
    Synthetic data generator using SDV (Synthetic Data Vault) library.

    Provides more sophisticated modeling including multi-table relationships
    and advanced statistical preservation.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        model_type: str = "gaussian_copula",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize SDV-based generator.

        Args:
            source_df: Source DataFrame
            model_type: SDV model type ('gaussian_copula', 'ctgan', 'copulagan', 'tvae')
            random_seed: Random seed
        """
        self.source_df = source_df
        self.model_type = model_type
        self.random_seed = random_seed
        self._model = None
        self._fitted = False

    def fit(self) -> "SDVSyntheticGenerator":
        """Fit the SDV model."""
        try:
            if self.model_type == "gaussian_copula":
                from sdv.single_table import GaussianCopulaSynthesizer

                self._model = GaussianCopulaSynthesizer(
                    metadata=self._create_metadata(),
                )
            elif self.model_type == "ctgan":
                from sdv.single_table import CTGANSynthesizer

                self._model = CTGANSynthesizer(
                    metadata=self._create_metadata(),
                    epochs=100,
                )
            elif self.model_type == "copulagan":
                from sdv.single_table import CopulaGANSynthesizer

                self._model = CopulaGANSynthesizer(
                    metadata=self._create_metadata(),
                )
            elif self.model_type == "tvae":
                from sdv.single_table import TVAESynthesizer

                self._model = TVAESynthesizer(
                    metadata=self._create_metadata(),
                    epochs=100,
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            logger.info(f"Fitting SDV {self.model_type} model...")
            self._model.fit(self.source_df)
            self._fitted = True
            logger.info("SDV model fitted successfully")

        except ImportError:
            logger.error("SDV not installed. Install with: pip install sdv")
            raise
        except Exception as e:
            logger.error(f"Error fitting SDV model: {e}")
            raise

        return self

    def _create_metadata(self):
        """Create SDV metadata from DataFrame."""
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.source_df)
        return metadata

    def generate(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic data using fitted SDV model."""
        if not self._fitted:
            self.fit()

        logger.info(f"Generating {n_rows} rows with SDV {self.model_type}")
        return self._model.sample(num_rows=n_rows)

    def evaluate(self, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate quality of synthetic data."""
        try:
            from sdv.evaluation.single_table import evaluate_quality

            report = evaluate_quality(
                real_data=self.source_df,
                synthetic_data=synthetic_df,
                metadata=self._create_metadata(),
            )

            return {
                "overall_score": report.get_score(),
                "properties": report.get_properties().to_dict(),
            }
        except Exception as e:
            logger.warning(f"Could not evaluate synthetic data: {e}")
            return {"error": str(e)}
