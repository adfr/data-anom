"""Application settings and configuration."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CDPSettings(BaseSettings):
    """Cloudera CDP connection settings."""

    model_config = SettingsConfigDict(env_prefix="CDP_")

    host: str = Field(default="localhost", description="CDP Impala host")
    port: int = Field(default=21050, description="CDP Impala port")
    database: str = Field(default="default", description="Default database")
    auth_mechanism: str = Field(default="PLAIN", description="Authentication mechanism")
    use_ssl: bool = Field(default=True, description="Use SSL connection")
    ca_cert: Optional[str] = Field(default=None, description="CA certificate path")
    user: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")


class IcebergSettings(BaseSettings):
    """Iceberg catalog settings."""

    model_config = SettingsConfigDict(env_prefix="ICEBERG_")

    catalog_name: str = Field(default="iceberg", description="Iceberg catalog name")
    catalog_type: str = Field(default="hive", description="Catalog type (hive, rest, glue)")
    warehouse: Optional[str] = Field(default=None, description="Warehouse location")
    uri: Optional[str] = Field(default=None, description="Catalog URI")


class SyntheticDataSettings(BaseSettings):
    """Synthetic data generation settings."""

    model_config = SettingsConfigDict(env_prefix="SYNTH_")

    default_sample_size: int = Field(default=1000, description="Default number of rows to generate")
    max_sample_size: int = Field(default=100000, description="Maximum rows to generate")
    preserve_correlations: bool = Field(default=True, description="Preserve column correlations")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="Synthetic Data Generator", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Sub-settings
    cdp: CDPSettings = Field(default_factory=CDPSettings)
    iceberg: IcebergSettings = Field(default_factory=IcebergSettings)
    synthetic: SyntheticDataSettings = Field(default_factory=SyntheticDataSettings)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
