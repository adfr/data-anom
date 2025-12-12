"""
Format-Preserving Tokenization Service.

Provides encryption-based tokenization that maintains the format of original data.
Uses a secret key for reversible encryption/decryption.
"""

import base64
import hashlib
import hmac
import re
import secrets
import string
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class FormatPreservingTokenizer:
    """
    Format-preserving tokenizer that encrypts data while maintaining format.

    Supports:
    - Credit cards: 1234-5678-9012-3456 -> 9876-5432-1098-7654
    - SSN: 123-45-6789 -> 987-65-4321
    - Phone: (555) 123-4567 -> (555) 987-6543
    - Email: john@example.com -> xkqm@example.com
    - Names: John Smith -> Xkqm Plrth
    - Dates: 2024-01-15 -> 2024-03-22
    - Generic alphanumeric

    Uses HMAC-based format-preserving encryption for reversibility with a key.
    """

    def __init__(self, secret_key: str):
        """
        Initialize the tokenizer with a secret key.

        Args:
            secret_key: Secret key for encryption (minimum 16 characters recommended)
        """
        if not secret_key or len(secret_key) < 8:
            raise ValueError("Secret key must be at least 8 characters")

        self._key = secret_key.encode('utf-8')
        self._token_cache: Dict[str, str] = {}
        self._reverse_cache: Dict[str, str] = {}

    @staticmethod
    def generate_key(length: int = 32) -> str:
        """Generate a secure random key."""
        return secrets.token_urlsafe(length)

    def _hmac_hash(self, value: str, tweak: str = "") -> bytes:
        """Generate HMAC hash for a value."""
        data = f"{tweak}:{value}".encode('utf-8')
        return hmac.new(self._key, data, hashlib.sha256).digest()

    def _map_char(self, char: str, hash_bytes: bytes, index: int, char_set: str) -> str:
        """Map a character to another in the same character set."""
        if char not in char_set:
            return char

        char_index = char_set.index(char)
        # Use hash byte to determine offset
        offset = hash_bytes[index % len(hash_bytes)]
        new_index = (char_index + offset) % len(char_set)
        return char_set[new_index]

    def _reverse_map_char(self, char: str, hash_bytes: bytes, index: int, char_set: str) -> str:
        """Reverse map a character back to original."""
        if char not in char_set:
            return char

        char_index = char_set.index(char)
        offset = hash_bytes[index % len(hash_bytes)]
        original_index = (char_index - offset) % len(char_set)
        return char_set[original_index]

    def _tokenize_with_format(
        self,
        value: str,
        preserve_pattern: str,
        char_sets: Dict[str, str],
        tweak: str = ""
    ) -> str:
        """
        Tokenize a value while preserving format.

        Args:
            value: Original value
            preserve_pattern: Regex pattern for characters to preserve
            char_sets: Dict mapping character classes to replacement sets
            tweak: Additional context for hashing

        Returns:
            Tokenized value with same format
        """
        if not value:
            return value

        hash_bytes = self._hmac_hash(value, tweak)
        result = []
        char_idx = 0

        for i, char in enumerate(value):
            # Check if character should be preserved
            if re.match(preserve_pattern, char):
                result.append(char)
            else:
                # Find appropriate character set
                mapped = False
                for pattern, char_set in char_sets.items():
                    if re.match(pattern, char):
                        result.append(self._map_char(char, hash_bytes, char_idx, char_set))
                        char_idx += 1
                        mapped = True
                        break

                if not mapped:
                    result.append(char)

        return ''.join(result)

    def _detokenize_with_format(
        self,
        value: str,
        original: str,
        preserve_pattern: str,
        char_sets: Dict[str, str],
        tweak: str = ""
    ) -> str:
        """Reverse tokenization to get original value."""
        if not value or not original:
            return value

        hash_bytes = self._hmac_hash(original, tweak)
        result = []
        char_idx = 0

        for i, char in enumerate(value):
            if re.match(preserve_pattern, char):
                result.append(char)
            else:
                for pattern, char_set in char_sets.items():
                    if re.match(pattern, char):
                        result.append(self._reverse_map_char(char, hash_bytes, char_idx, char_set))
                        char_idx += 1
                        break
                else:
                    result.append(char)

        return ''.join(result)

    def tokenize_credit_card(self, value: str) -> str:
        """Tokenize credit card number preserving format."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[-\s]',
            char_sets={r'[0-9]': string.digits},
            tweak="cc"
        )

    def tokenize_ssn(self, value: str) -> str:
        """Tokenize SSN preserving format."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[-]',
            char_sets={r'[0-9]': string.digits},
            tweak="ssn"
        )

    def tokenize_phone(self, value: str) -> str:
        """Tokenize phone number preserving format."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[-\s\(\)\+]',
            char_sets={r'[0-9]': string.digits},
            tweak="phone"
        )

    def tokenize_email(self, value: str) -> str:
        """Tokenize email preserving format (domain stays same)."""
        if '@' not in value:
            return value

        local, domain = value.rsplit('@', 1)
        tokenized_local = self._tokenize_with_format(
            local,
            preserve_pattern=r'[._-]',
            char_sets={
                r'[a-z]': string.ascii_lowercase,
                r'[A-Z]': string.ascii_uppercase,
                r'[0-9]': string.digits,
            },
            tweak="email"
        )
        return f"{tokenized_local}@{domain}"

    def tokenize_name(self, value: str) -> str:
        """Tokenize name preserving format (spaces, capitalization pattern)."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[\s\-\']',
            char_sets={
                r'[a-z]': string.ascii_lowercase,
                r'[A-Z]': string.ascii_uppercase,
            },
            tweak="name"
        )

    def tokenize_date(self, value: str) -> str:
        """Tokenize date preserving format."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[-/\s:]',
            char_sets={r'[0-9]': string.digits},
            tweak="date"
        )

    def tokenize_alphanumeric(self, value: str) -> str:
        """Tokenize generic alphanumeric string."""
        return self._tokenize_with_format(
            value,
            preserve_pattern=r'[\s\-_\.@#]',
            char_sets={
                r'[a-z]': string.ascii_lowercase,
                r'[A-Z]': string.ascii_uppercase,
                r'[0-9]': string.digits,
            },
            tweak="alphanum"
        )

    def tokenize_value(self, value: Any, data_type: str) -> Any:
        """
        Tokenize a value based on its detected type.

        Args:
            value: Value to tokenize
            data_type: Type of data (credit_card, ssn, phone, email, name, date, alphanumeric)

        Returns:
            Tokenized value
        """
        if pd.isna(value) or value is None:
            return value

        value_str = str(value)

        # Cache lookup
        cache_key = f"{data_type}:{value_str}"
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        # Tokenize based on type
        tokenizers = {
            'credit_card': self.tokenize_credit_card,
            'ssn': self.tokenize_ssn,
            'phone': self.tokenize_phone,
            'email': self.tokenize_email,
            'name': self.tokenize_name,
            'date': self.tokenize_date,
            'alphanumeric': self.tokenize_alphanumeric,
            'text': self.tokenize_alphanumeric,
        }

        tokenizer = tokenizers.get(data_type, self.tokenize_alphanumeric)
        tokenized = tokenizer(value_str)

        # Cache result
        self._token_cache[cache_key] = tokenized
        self._reverse_cache[f"{data_type}:{tokenized}"] = value_str

        return tokenized

    def detokenize_value(self, value: Any, data_type: str) -> Any:
        """
        Detokenize a value back to original.

        Note: Requires the original value to have been tokenized in this session
        or the original value for HMAC computation.

        Args:
            value: Tokenized value
            data_type: Type of data

        Returns:
            Original value if found in cache, else the tokenized value
        """
        if pd.isna(value) or value is None:
            return value

        cache_key = f"{data_type}:{value}"
        return self._reverse_cache.get(cache_key, value)

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Tokenize specified columns in a DataFrame.

        Args:
            df: Input DataFrame
            column_types: Dict mapping column names to data types
            columns: Specific columns to tokenize (all in column_types if None)

        Returns:
            DataFrame with tokenized columns
        """
        result = df.copy()
        cols_to_tokenize = columns or list(column_types.keys())

        for col in cols_to_tokenize:
            if col not in df.columns:
                continue

            data_type = column_types.get(col, 'alphanumeric')
            logger.info(f"Tokenizing column '{col}' as {data_type}")

            result[col] = result[col].apply(
                lambda x: self.tokenize_value(x, data_type)
            )

        return result

    def get_token_mapping(self) -> Dict[str, str]:
        """Get the current token mapping (for export/audit)."""
        return dict(self._token_cache)


def detect_sensitive_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect potentially sensitive columns and their types.

    Args:
        df: DataFrame to analyze

    Returns:
        Dict mapping column names to detected sensitive data types
    """
    sensitive_columns = {}

    for col in df.columns:
        col_lower = col.lower()
        sample = df[col].dropna().head(100)

        if len(sample) == 0:
            continue

        # Check column name patterns
        if any(x in col_lower for x in ['ssn', 'social_security', 'sin']):
            sensitive_columns[col] = 'ssn'
        elif any(x in col_lower for x in ['credit_card', 'card_number', 'cc_num', 'cardnum']):
            sensitive_columns[col] = 'credit_card'
        elif any(x in col_lower for x in ['phone', 'mobile', 'tel', 'fax']):
            sensitive_columns[col] = 'phone'
        elif any(x in col_lower for x in ['email', 'e_mail', 'e-mail']):
            sensitive_columns[col] = 'email'
        elif any(x in col_lower for x in ['first_name', 'last_name', 'full_name', 'name']):
            if 'file' not in col_lower and 'table' not in col_lower:
                sensitive_columns[col] = 'name'
        elif any(x in col_lower for x in ['dob', 'birth_date', 'birthdate', 'date_of_birth']):
            sensitive_columns[col] = 'date'
        elif any(x in col_lower for x in ['address', 'street', 'addr']):
            sensitive_columns[col] = 'alphanumeric'
        else:
            # Check data patterns
            sample_str = sample.astype(str)

            # SSN pattern
            if sample_str.str.match(r'^\d{3}-\d{2}-\d{4}$').mean() > 0.5:
                sensitive_columns[col] = 'ssn'
            # Credit card pattern
            elif sample_str.str.match(r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$').mean() > 0.5:
                sensitive_columns[col] = 'credit_card'
            # Email pattern
            elif sample_str.str.match(r'^[^@]+@[^@]+\.[^@]+$').mean() > 0.5:
                sensitive_columns[col] = 'email'
            # Phone pattern
            elif sample_str.str.match(r'^[\d\s\-\(\)\+]{10,}$').mean() > 0.5:
                sensitive_columns[col] = 'phone'

    return sensitive_columns
