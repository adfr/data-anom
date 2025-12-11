"""Cloudera AI (CML) native data lake connector using Spark and Iceberg."""

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class CMLDataLakeConnector:
    """
    Native connector for Cloudera AI/CML environment.

    Uses SparkSession with Iceberg catalog for direct data lake access.
    No external credentials needed - uses CML's built-in authentication.
    """

    def __init__(
        self,
        catalog_name: str = "spark_catalog",
        warehouse: Optional[str] = None,
    ):
        """
        Initialize the CML Data Lake connector.

        Args:
            catalog_name: Iceberg catalog name (default: spark_catalog)
            warehouse: Optional warehouse location override
        """
        self.catalog_name = catalog_name
        self.warehouse = warehouse
        self._spark = None
        self._initialized = False

    def _get_spark(self):
        """Get or create SparkSession with Iceberg support."""
        if self._spark is not None:
            return self._spark

        try:
            from pyspark.sql import SparkSession

            builder = SparkSession.builder \
                .appName("SyntheticDataGenerator") \
                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
                .config(f"spark.sql.catalog.{self.catalog_name}", "org.apache.iceberg.spark.SparkCatalog") \
                .config(f"spark.sql.catalog.{self.catalog_name}.type", "hive") \
                .config("spark.sql.iceberg.handle-timestamp-without-timezone", "true")

            if self.warehouse:
                builder = builder.config(
                    f"spark.sql.catalog.{self.catalog_name}.warehouse",
                    self.warehouse
                )

            # Enable Hive support for metastore access
            self._spark = builder.enableHiveSupport().getOrCreate()

            # Set catalog
            self._spark.sql(f"USE {self.catalog_name}")

            logger.info(f"SparkSession initialized with catalog: {self.catalog_name}")
            self._initialized = True

            return self._spark

        except Exception as e:
            logger.error(f"Failed to initialize SparkSession: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if Spark session is active."""
        return self._initialized and self._spark is not None

    def list_databases(self) -> List[str]:
        """
        List available databases in the data lake.

        Returns:
            List of database names
        """
        spark = self._get_spark()
        databases = spark.sql("SHOW DATABASES").collect()
        return [row.namespace for row in databases]

    def list_tables(self, database: str = "default") -> List[str]:
        """
        List tables in a database.

        Args:
            database: Database name

        Returns:
            List of table names
        """
        spark = self._get_spark()
        tables = spark.sql(f"SHOW TABLES IN {database}").collect()
        return [row.tableName for row in tables]

    def get_table_schema(self, table: str, database: str = "default") -> List[Dict[str, Any]]:
        """
        Get schema information for a table.

        Args:
            table: Table name
            database: Database name

        Returns:
            List of column definitions
        """
        spark = self._get_spark()
        full_table = f"{database}.{table}"

        df = spark.table(full_table)
        schema = df.schema

        columns = []
        for field in schema.fields:
            columns.append({
                "name": field.name,
                "type": str(field.dataType),
                "nullable": field.nullable,
                "comment": field.metadata.get("comment") if field.metadata else None,
            })

        return columns

    def read_table(
        self,
        table: str,
        database: str = "default",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        where: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read data from an Iceberg table.

        Args:
            table: Table name
            database: Database name
            columns: Specific columns to select (all if None)
            limit: Maximum number of rows to read
            where: Optional filter condition

        Returns:
            Pandas DataFrame with the table data
        """
        spark = self._get_spark()
        full_table = f"{database}.{table}"

        logger.info(f"Reading table: {full_table}")

        # Start with the table
        df = spark.table(full_table)

        # Select columns
        if columns:
            df = df.select(*columns)

        # Apply filter
        if where:
            df = df.filter(where)

        # Apply limit
        if limit:
            df = df.limit(limit)

        # Convert to Pandas
        return df.toPandas()

    def read_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query string

        Returns:
            Pandas DataFrame with query results
        """
        spark = self._get_spark()
        logger.info(f"Executing query: {query[:100]}...")

        df = spark.sql(query)
        return df.toPandas()

    def get_table_info(self, table: str, database: str = "default") -> Dict[str, Any]:
        """
        Get detailed table information including Iceberg metadata.

        Args:
            table: Table name
            database: Database name

        Returns:
            Dictionary with table information
        """
        spark = self._get_spark()
        full_table = f"{database}.{table}"

        # Get basic info
        schema = self.get_table_schema(table, database)

        # Get row count (approximate for large tables)
        count_df = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table}")
        row_count = count_df.collect()[0].cnt

        # Try to get Iceberg-specific metadata
        iceberg_info = {}
        try:
            history = spark.sql(f"SELECT * FROM {full_table}.history LIMIT 10").collect()
            iceberg_info["snapshots"] = len(history)
            if history:
                iceberg_info["latest_snapshot"] = str(history[0].made_current_at)
        except Exception:
            pass  # Table might not be Iceberg

        return {
            "database": database,
            "table": table,
            "full_name": full_table,
            "row_count": row_count,
            "column_count": len(schema),
            "columns": schema,
            "iceberg": iceberg_info,
        }

    def get_sample(
        self,
        table: str,
        database: str = "default",
        n: int = 1000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get a random sample from a table.

        Args:
            table: Table name
            database: Database name
            n: Number of rows to sample
            seed: Random seed for reproducibility

        Returns:
            Pandas DataFrame with sampled data
        """
        spark = self._get_spark()
        full_table = f"{database}.{table}"

        df = spark.table(full_table)

        # Get approximate count for sampling fraction
        total_count = df.count()

        if total_count <= n:
            return df.toPandas()

        fraction = min(1.0, (n * 1.5) / total_count)  # Oversample slightly

        if seed is not None:
            sampled = df.sample(fraction=fraction, seed=seed).limit(n)
        else:
            sampled = df.sample(fraction=fraction).limit(n)

        return sampled.toPandas()

    def write_table(
        self,
        df: pd.DataFrame,
        table: str,
        database: str = "default",
        mode: str = "overwrite",
        format: str = "iceberg",
    ) -> None:
        """
        Write a DataFrame to the data lake.

        Args:
            df: Pandas DataFrame to write
            table: Target table name
            database: Database name
            mode: Write mode ('overwrite', 'append', 'error')
            format: Table format ('iceberg', 'parquet', 'delta')
        """
        spark = self._get_spark()
        full_table = f"{database}.{table}"

        logger.info(f"Writing {len(df)} rows to {full_table}")

        # Convert Pandas to Spark DataFrame
        spark_df = spark.createDataFrame(df)

        # Write to table
        spark_df.write \
            .format(format) \
            .mode(mode) \
            .saveAsTable(full_table)

        logger.info(f"Successfully wrote to {full_table}")

    def close(self):
        """Stop the Spark session."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            self._initialized = False
            logger.info("SparkSession stopped")


class CMLDataConnection:
    """
    Helper class for CML Data Connections.

    Uses CML's built-in data connection feature for simplified access.
    """

    def __init__(self, connection_name: str):
        """
        Initialize using a CML Data Connection.

        Args:
            connection_name: Name of the CML Data Connection
        """
        self.connection_name = connection_name
        self._connection = None

    def connect(self):
        """Establish connection using CML Data Connections API."""
        try:
            import cml.data_v1 as cmldata

            self._connection = cmldata.get_connection(self.connection_name)
            logger.info(f"Connected via CML Data Connection: {self.connection_name}")
            return self._connection

        except ImportError:
            logger.warning("CML data module not available. Running outside CML?")
            raise
        except Exception as e:
            logger.error(f"Failed to get CML Data Connection: {e}")
            raise

    def read_sql(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query using the data connection.

        Args:
            query: SQL query string

        Returns:
            Pandas DataFrame with results
        """
        if self._connection is None:
            self.connect()

        return self._connection.get_pandas_dataframe(query)


def get_connector(use_cml: bool = True, **kwargs):
    """
    Factory function to get the appropriate connector.

    Args:
        use_cml: Whether to use CML native connector
        **kwargs: Additional arguments for the connector

    Returns:
        Data connector instance
    """
    if use_cml:
        try:
            connector = CMLDataLakeConnector(**kwargs)
            connector._get_spark()  # Test connection
            return connector
        except Exception as e:
            logger.warning(f"CML connector failed: {e}. Falling back to mock.")

    # Fallback to mock connector for local development
    from app.services.cdp_connector import MockCDPConnector
    return MockCDPConnector()
