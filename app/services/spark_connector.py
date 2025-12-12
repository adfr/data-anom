"""Spark connector for Cloudera Data Lake access.

Uses CML Data Connection to get a pre-configured Spark session
for accessing Data Lake tables including Iceberg.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class SparkConnector:
    """
    Spark-based connector for Cloudera Data Lake via CML Data Connection.

    Uses CML's data connection to get a properly configured Spark session
    with access to Hive metastore and Iceberg tables.
    """

    def __init__(self, connection_name: Optional[str] = None):
        """
        Initialize the Spark connector.

        Args:
            connection_name: CML Data Connection name (required for Data Lake)
        """
        self.connection_name = connection_name
        self._spark = None
        self._connection = None
        self._initialized = False
        self._connection_info = {}

    def _get_spark(self):
        """Get Spark session from CML Data Connection."""
        if self._spark is not None:
            return self._spark

        try:
            import cml.data_v1 as cmldata
            import os

            # Get connection name from parameter or environment
            conn_name = self.connection_name or os.environ.get('CML_CONNECTION_NAME')

            if not conn_name:
                # Try to list available connections
                if hasattr(cmldata, 'list_connections'):
                    connections = cmldata.list_connections()
                    if connections:
                        conn_name = connections[0]
                        logger.info(f"Auto-selected CML connection: {conn_name}")

            if not conn_name:
                raise ValueError(
                    "No CML Data Connection specified. "
                    "Set CML_CONNECTION_NAME environment variable."
                )

            # Get connection and Spark session
            logger.info(f"Getting Spark session from CML connection: {conn_name}")
            self._connection = cmldata.get_connection(conn_name)
            self._spark = self._connection.get_spark_session()

            # Configure for Iceberg timestamp handling
            self._spark.conf.set("spark.sql.iceberg.handle-timestamp-without-timezone", "true")

            self.connection_name = conn_name
            self._initialized = True
            self._connection_info = {
                "connection_name": conn_name,
                "spark_version": self._spark.version,
                "app_id": self._spark.sparkContext.applicationId,
            }

            logger.info(f"Spark session created: {self._spark.version}")
            return self._spark

        except ImportError as e:
            logger.error("CML data module not available")
            raise RuntimeError(f"CML not available: {e}")
        except Exception as e:
            logger.error(f"Failed to get Spark session: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if Spark session is active."""
        return self._initialized and self._spark is not None

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current connection (without initializing Spark)."""
        import os
        conn_name = self.connection_name or os.environ.get('CML_CONNECTION_NAME') or "spark"

        return {
            "name": conn_name,
            "type": "spark",
            "type_label": "Spark (Data Lake)",
            "available_connections": [conn_name],
            "details": self._connection_info if self._initialized else {"status": "ready"},
        }

    def list_databases(self) -> List[str]:
        """
        List available databases from Hive metastore.

        Returns:
            List of database names
        """
        spark = self._get_spark()

        try:
            # Limit to 100 databases for performance
            df = spark.sql("SHOW DATABASES").limit(100)
            # Get the first column regardless of name (namespace, databaseName, etc.)
            return [row[0] for row in df.collect()]
        except Exception as e:
            logger.error(f"Error listing databases: {e}")
            raise

    def list_tables(self, database: str = "default") -> List[str]:
        """
        List tables in a database.

        Args:
            database: Database name

        Returns:
            List of table names
        """
        spark = self._get_spark()

        try:
            # Limit to 200 tables for performance
            df = spark.sql(f"SHOW TABLES IN {database}").limit(200)
            # tableName is usually second column (first is database)
            return [row[1] if len(row) > 1 else row[0] for row in df.collect()]
        except Exception as e:
            logger.error(f"Error listing tables in {database}: {e}")
            raise

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

        try:
            df = spark.sql(f"DESCRIBE {database}.{table}")

            columns = []
            for row in df.collect():
                # Use index access for compatibility
                col_name = row[0] if len(row) > 0 else None
                col_type = row[1] if len(row) > 1 else None
                col_comment = row[2] if len(row) > 2 else None

                # Skip partition info rows and empty rows
                if col_name and not str(col_name).startswith('#') and str(col_name).strip():
                    columns.append({
                        "name": str(col_name).strip(),
                        "type": str(col_type).strip() if col_type else "unknown",
                        "comment": str(col_comment).strip() if col_comment and str(col_comment).strip() else None,
                    })

            return columns
        except Exception as e:
            logger.error(f"Error getting schema for {database}.{table}: {e}")
            raise

    def read_table(
        self,
        table: str,
        database: str = "default",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        where: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read data from a table.

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

        # Build query
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {database}.{table}"

        if where:
            query += f" WHERE {where}"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing Spark SQL: {query}")

        try:
            spark_df = spark.sql(query)
            return spark_df.toPandas()
        except Exception as e:
            logger.error(f"Error reading table {database}.{table}: {e}")
            raise

    def read_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query string

        Returns:
            Pandas DataFrame with query results
        """
        spark = self._get_spark()
        logger.info(f"Executing Spark SQL: {query[:100]}...")

        try:
            spark_df = spark.sql(query)
            return spark_df.toPandas()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_table_info(self, table: str, database: str = "default") -> Dict[str, Any]:
        """
        Get detailed table information.

        Args:
            table: Table name
            database: Database name

        Returns:
            Dictionary with table information
        """
        spark = self._get_spark()

        # Get schema
        schema = self.get_table_schema(table, database)

        # Get row count
        try:
            count_df = spark.sql(f"SELECT COUNT(*) as cnt FROM {database}.{table}")
            row_count = count_df.collect()[0][0]
        except Exception:
            row_count = None

        return {
            "database": database,
            "table": table,
            "full_name": f"{database}.{table}",
            "row_count": row_count,
            "column_count": len(schema),
            "columns": schema,
            "connection": "spark",
            "connection_type": "spark",
        }

    def get_sample(
        self,
        table: str,
        database: str = "default",
        n: int = 200,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get a sample from a table.

        Args:
            table: Table name
            database: Database name
            n: Number of rows to sample (default 200)
            seed: Random seed (unused, for API compatibility)

        Returns:
            Pandas DataFrame with sampled data
        """
        spark = self._get_spark()

        # Simple LIMIT query - fast and reliable
        query = f"SELECT * FROM {database}.{table} LIMIT {n}"
        logger.info(f"Executing: {query}")

        try:
            spark_df = spark.sql(query)
            return spark_df.toPandas()
        except Exception as e:
            logger.error(f"Error sampling {database}.{table}: {e}")
            raise

    def close(self):
        """Stop the Spark session."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            self._initialized = False
            logger.info("Spark session stopped")


def get_spark_connector(connection_name: Optional[str] = None) -> SparkConnector:
    """
    Factory function to get the Spark connector.

    Args:
        connection_name: CML Data Connection name (optional, will auto-detect)

    Returns:
        SparkConnector instance
    """
    connector = SparkConnector(connection_name=connection_name)
    connector._get_spark()  # Initialize Spark session
    return connector
