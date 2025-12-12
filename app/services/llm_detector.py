"""
LLM-based smart column detection for sensitive data.

Uses Anthropic Claude to intelligently detect sensitive data types
in column names and sample values.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class LLMColumnDetector:
    """
    Uses LLM to intelligently detect sensitive data columns and their types.

    Supports detection of:
    - Credit cards
    - SSN / Social Security Numbers
    - Phone numbers
    - Email addresses
    - Names (first, last, full)
    - Dates of birth
    - Addresses
    - Custom sensitive patterns
    """

    SUPPORTED_TYPES = [
        "credit_card",
        "ssn",
        "phone",
        "email",
        "name",
        "date",
        "address",
        "alphanumeric",
        "not_sensitive"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM detector.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not provided. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def _prepare_column_info(self, df: pd.DataFrame, max_samples: int = 10) -> List[Dict[str, Any]]:
        """Prepare column information for LLM analysis."""
        columns_info = []

        for col in df.columns:
            sample = df[col].dropna().head(max_samples)
            sample_values = [str(v) for v in sample.tolist()]

            columns_info.append({
                "column_name": col,
                "dtype": str(df[col].dtype),
                "sample_values": sample_values,
                "null_count": int(df[col].isna().sum()),
                "total_count": len(df),
            })

        return columns_info

    def detect_sensitive_columns(
        self,
        df: pd.DataFrame,
        additional_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Use LLM to detect sensitive columns and classify their types.

        Args:
            df: DataFrame to analyze
            additional_context: Optional context about the data

        Returns:
            Dict mapping column names to detected sensitive data types
        """
        client = self._get_client()
        columns_info = self._prepare_column_info(df)

        prompt = f"""Analyze the following DataFrame columns and identify which contain sensitive/PII data that should be tokenized or anonymized.

For each column, classify it as one of these types:
- credit_card: Credit card numbers (16 digits, possibly with dashes/spaces)
- ssn: Social Security Numbers (###-##-#### format or 9 digits)
- phone: Phone numbers (various formats)
- email: Email addresses
- name: Person names (first name, last name, or full name)
- date: Dates that could be sensitive (like birth dates)
- address: Physical addresses
- alphanumeric: Other sensitive alphanumeric identifiers
- not_sensitive: Not sensitive data, do not tokenize

Column information:
```json
{json.dumps(columns_info, indent=2)}
```

{f"Additional context: {additional_context}" if additional_context else ""}

IMPORTANT: Only mark columns as sensitive if they truly contain PII or sensitive data.
Common non-sensitive data like product categories, status codes, or generic descriptions should be marked as "not_sensitive".

Respond ONLY with a JSON object mapping column names to their detected type.
Example: {{"email_address": "email", "customer_name": "name", "product_id": "not_sensitive"}}

JSON response:"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)

            # Filter to only include sensitive columns
            sensitive_columns = {
                col: dtype
                for col, dtype in result.items()
                if dtype != "not_sensitive" and dtype in self.SUPPORTED_TYPES
            }

            logger.info(f"LLM detected {len(sensitive_columns)} sensitive columns: {list(sensitive_columns.keys())}")
            return sensitive_columns

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            # Fall back to rule-based detection
            from app.services.tokenizer import detect_sensitive_columns
            return detect_sensitive_columns(df)
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            # Fall back to rule-based detection
            from app.services.tokenizer import detect_sensitive_columns
            return detect_sensitive_columns(df)

    def explain_detection(
        self,
        df: pd.DataFrame,
        detected_columns: Dict[str, str]
    ) -> str:
        """
        Get an explanation of why columns were classified as sensitive.

        Args:
            df: DataFrame analyzed
            detected_columns: Previously detected sensitive columns

        Returns:
            Human-readable explanation
        """
        client = self._get_client()
        columns_info = self._prepare_column_info(df)

        # Filter to only detected columns
        detected_info = [
            info for info in columns_info
            if info["column_name"] in detected_columns
        ]

        prompt = f"""Explain why the following columns were classified as sensitive PII data:

Classifications:
{json.dumps(detected_columns, indent=2)}

Column details:
{json.dumps(detected_info, indent=2)}

Provide a brief explanation for each column (1-2 sentences each) explaining why it was classified as that type.
Format as a bulleted list."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Failed to get explanation: {e}")
            return "Unable to generate explanation."


def is_llm_available() -> bool:
    """Check if LLM detection is available (API key is set)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def detect_with_llm(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    additional_context: Optional[str] = None
) -> Dict[str, str]:
    """
    Convenience function for LLM-based detection.

    Args:
        df: DataFrame to analyze
        api_key: Optional API key (uses env var if not provided)
        additional_context: Optional context about the data

    Returns:
        Dict mapping column names to sensitive data types
    """
    detector = LLMColumnDetector(api_key=api_key)
    return detector.detect_sensitive_columns(df, additional_context)
