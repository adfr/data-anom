"""Cloudera AI (CML) native data lake connector using CML Data Connections."""

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class CMLDataLakeConnector:
    """
    Native connector for Cloudera AI/CML environment using CML Data Connections.

    Uses CML's built-in data connection feature for simplified, secure access
    to the Cloudera Data Lake (Hive, Impala, Iceberg tables).
    """

    def __init__(self, connection_name: Optional[str] = None):
        """
        Initialize the CML Data Lake connector.

        Args:
            connection_name: Name of the CML Data Connection (optional, will auto-detect)
        """
        self.connection_name = connection_name
        self._connection = None
        self._connection_type = None
        self._initialized = False
        self._available_connections = []

    def _get_connection(self):
        """Get or create CML Data Connection."""
        if self._connection is not None:
            return self._connection

        try:
            import cml.data_v1 as cmldata

            # List available connections
            self._available_connections = cmldata.list_connections()
            logger.info(f"Available CML Data Connections: {self._available_connections}")

            if not self._available_connections:
                raise ValueError("No CML Data Connections configured. Please add a data connection in the CML project settings.")

            # Use specified connection or first available
            if self.connection_name:
                conn_name = self.connection_name
            else:
                # Try to find a Hive/Impala connection
                for conn in self._available_connections:
                    conn_lower = conn.lower()
                    if any(x in conn_lower for x in ['hive', 'impala', 'warehouse', 'datalake', 'lake']):
                        conn_name = conn
                        break
                else:
                    conn_name = self._available_connections[0]

            logger.info(f"Using CML Data Connection: {conn_name}")
            self._connection = cmldata.get_connection(conn_name)
            self.connection_name = conn_name
            self._initialized = True

            return self._connection

        except ImportError as e:
            logger.error("CML data module not available. Are you running in CML?")
            logger.error(f"Import error: {e}")
            raise RuntimeError(
                "CML Data Connections not available. "
                "Make sure you're running in Cloudera Machine Learning "
                "and have data connections configured."
            )
        except Exception as e:
            logger.error(f"Failed to initialize CML Data Connection: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._initialized and self._connection is not None

    def get_available_connections(self) -> List[str]:
        """Get list of available CML Data Connections."""
        try:
            import cml.data_v1 as cmldata
            return cmldata.list_connections()
        except Exception:
            return []

    def list_databases(self) -> List[str]:
        """
        List available databases.

        Returns:
            List of database names
        """
        conn = self._get_connection()
        query = "SHOW DATABASES"

        try:
            df = conn.get_pandas_dataframe(query)
            # Column name varies by connection type
            if 'database_name' in df.columns:
                return df['database_name'].tolist()
            elif 'namespace' in df.columns:
                return df['namespace'].tolist()
            elif 'databaseName' in df.columns:
                return df['databaseName'].tolist()
            else:
                return df.iloc[:, 0].tolist()
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
        conn = self._get_connection()
        query = f"SHOW TABLES IN {database}"

        try:
            df = conn.get_pandas_dataframe(query)
            # Column name varies
            if 'tableName' in df.columns:
                return df['tableName'].tolist()
            elif 'tab_name' in df.columns:
                return df['tab_name'].tolist()
            elif 'table_name' in df.columns:
                return df['table_name'].tolist()
            else:
                # Usually second column contains table name
                return df.iloc[:, -1].tolist() if len(df.columns) > 1 else df.iloc[:, 0].tolist()
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
        conn = self._get_connection()
        query = f"DESCRIBE {database}.{table}"

        try:
            df = conn.get_pandas_dataframe(query)

            columns = []
            for _, row in df.iterrows():
                col_name = row.iloc[0] if len(row) > 0 else None
                col_type = row.iloc[1] if len(row) > 1 else None
                col_comment = row.iloc[2] if len(row) > 2 else None

                # Skip partition info rows
                if col_name and not col_name.startswith('#') and col_name.strip():
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
        conn = self._get_connection()

        # Build query
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {database}.{table}"

        if where:
            query += f" WHERE {where}"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing query: {query}")

        try:
            return conn.get_pandas_dataframe(query)
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
        conn = self._get_connection()
        logger.info(f"Executing query: {query[:100]}...")

        try:
            return conn.get_pandas_dataframe(query)
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
        # Get schema
        schema = self.get_table_schema(table, database)

        # Get row count
        try:
            conn = self._get_connection()
            count_df = conn.get_pandas_dataframe(f"SELECT COUNT(*) as cnt FROM {database}.{table}")
            row_count = int(count_df.iloc[0, 0])
        except Exception:
            row_count = None

        return {
            "database": database,
            "table": table,
            "full_name": f"{database}.{table}",
            "row_count": row_count,
            "column_count": len(schema),
            "columns": schema,
            "connection": self.connection_name,
        }

    def get_sample(
        self,
        table: str,
        database: str = "default",
        n: int = 1000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get a sample from a table.

        Args:
            table: Table name
            database: Database name
            n: Number of rows to sample
            seed: Random seed (not used with SQL LIMIT)

        Returns:
            Pandas DataFrame with sampled data
        """
        # Simple limit-based sampling (TABLESAMPLE not always available)
        return self.read_table(table, database, limit=n)

    def close(self):
        """Close the connection."""
        self._connection = None
        self._initialized = False
        logger.info("CML Data Connection closed")


def get_connector(connection_name: Optional[str] = None):
    """
    Factory function to get the CML connector.

    Args:
        connection_name: Optional specific connection name

    Returns:
        CMLDataLakeConnector instance
    """
    try:
        connector = CMLDataLakeConnector(connection_name=connection_name)
        connector._get_connection()  # Test connection
        return connector
    except Exception as e:
        logger.warning(f"CML connector failed: {e}. Falling back to mock.")
        from app.services.cdp_connector import MockCDPConnector
        mock = MockCDPConnector()
        mock.connect()
        return mock
