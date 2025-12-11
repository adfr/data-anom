"""Cloudera CDP and Iceberg data connector."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import CDPSettings, IcebergSettings


class CDPConnector:
    """
    Connector for Cloudera CDP with Iceberg table support.

    Supports connecting to CDP via Impala and reading Iceberg tables.
    """

    def __init__(
        self,
        cdp_settings: Optional[CDPSettings] = None,
        iceberg_settings: Optional[IcebergSettings] = None,
    ):
        """
        Initialize the CDP connector.

        Args:
            cdp_settings: CDP connection settings
            iceberg_settings: Iceberg catalog settings
        """
        self.cdp_settings = cdp_settings or CDPSettings()
        self.iceberg_settings = iceberg_settings or IcebergSettings()
        self._connection = None
        self._cursor = None

    def connect(self) -> bool:
        """
        Establish connection to CDP Impala.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from impala.dbapi import connect

            conn_params = {
                "host": self.cdp_settings.host,
                "port": self.cdp_settings.port,
                "database": self.cdp_settings.database,
                "auth_mechanism": self.cdp_settings.auth_mechanism,
                "use_ssl": self.cdp_settings.use_ssl,
            }

            if self.cdp_settings.user:
                conn_params["user"] = self.cdp_settings.user
            if self.cdp_settings.password:
                conn_params["password"] = self.cdp_settings.password
            if self.cdp_settings.ca_cert:
                conn_params["ca_cert"] = self.cdp_settings.ca_cert

            self._connection = connect(**conn_params)
            self._cursor = self._connection.cursor()
            logger.info(f"Connected to CDP at {self.cdp_settings.host}:{self.cdp_settings.port}")
            return True

        except ImportError:
            logger.error("impyla package not installed. Install with: pip install impyla")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to CDP: {e}")
            return False

    def disconnect(self):
        """Close the CDP connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Disconnected from CDP")

    def is_connected(self) -> bool:
        """Check if connected to CDP."""
        return self._connection is not None

    def list_databases(self) -> List[str]:
        """
        List available databases.

        Returns:
            List of database names
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        self._cursor.execute("SHOW DATABASES")
        return [row[0] for row in self._cursor.fetchall()]

    def list_tables(self, database: Optional[str] = None) -> List[str]:
        """
        List tables in a database.

        Args:
            database: Database name (uses default if not specified)

        Returns:
            List of table names
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        db = database or self.cdp_settings.database
        self._cursor.execute(f"SHOW TABLES IN {db}")
        return [row[0] for row in self._cursor.fetchall()]

    def get_table_schema(self, table: str, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get schema information for a table.

        Args:
            table: Table name
            database: Database name (uses default if not specified)

        Returns:
            List of column definitions with name, type, and description
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        db = database or self.cdp_settings.database
        self._cursor.execute(f"DESCRIBE {db}.{table}")

        columns = []
        for row in self._cursor.fetchall():
            columns.append(
                {
                    "name": row[0],
                    "type": row[1],
                    "comment": row[2] if len(row) > 2 else None,
                }
            )
        return columns

    def read_table(
        self,
        table: str,
        database: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        where: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read data from an Iceberg table.

        Args:
            table: Table name
            database: Database name (uses default if not specified)
            columns: Specific columns to select (all if None)
            limit: Maximum number of rows to read
            where: Optional WHERE clause

        Returns:
            DataFrame with the table data
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        db = database or self.cdp_settings.database
        cols = ", ".join(columns) if columns else "*"

        query = f"SELECT {cols} FROM {db}.{table}"

        if where:
            query += f" WHERE {where}"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing query: {query}")
        self._cursor.execute(query)

        # Get column names from cursor description
        col_names = [desc[0] for desc in self._cursor.description]
        data = self._cursor.fetchall()

        return pd.DataFrame(data, columns=col_names)

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        logger.info(f"Executing custom query: {query[:100]}...")
        self._cursor.execute(query)

        col_names = [desc[0] for desc in self._cursor.description]
        data = self._cursor.fetchall()

        return pd.DataFrame(data, columns=col_names)

    def get_table_stats(self, table: str, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a table.

        Args:
            table: Table name
            database: Database name

        Returns:
            Dictionary with table statistics
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to CDP")

        db = database or self.cdp_settings.database

        # Get row count
        self._cursor.execute(f"SELECT COUNT(*) FROM {db}.{table}")
        row_count = self._cursor.fetchone()[0]

        # Get schema
        schema = self.get_table_schema(table, database)

        return {
            "database": db,
            "table": table,
            "row_count": row_count,
            "column_count": len(schema),
            "columns": schema,
        }


class MockCDPConnector(CDPConnector):
    """
    Mock CDP connector for demo/testing purposes.

    Simulates CDP connection without actual connectivity.
    """

    def __init__(self, *args, **kwargs):
        """Initialize mock connector."""
        super().__init__(*args, **kwargs)
        self._mock_connected = False
        self._mock_data = {}

    def connect(self) -> bool:
        """Simulate connection."""
        self._mock_connected = True
        logger.info("Mock CDP connection established")
        return True

    def disconnect(self):
        """Simulate disconnection."""
        self._mock_connected = False
        logger.info("Mock CDP connection closed")

    def is_connected(self) -> bool:
        """Check mock connection status."""
        return self._mock_connected

    def list_databases(self) -> List[str]:
        """Return mock databases."""
        return ["default", "sales_db", "customer_db", "analytics"]

    def list_tables(self, database: Optional[str] = None) -> List[str]:
        """Return mock tables."""
        tables = {
            "default": ["sample_data", "test_table"],
            "sales_db": ["transactions", "products", "customers"],
            "customer_db": ["users", "profiles", "preferences"],
            "analytics": ["events", "metrics", "aggregates"],
        }
        db = database or "default"
        return tables.get(db, [])

    def get_table_schema(self, table: str, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock schema."""
        schemas = {
            "transactions": [
                {"name": "transaction_id", "type": "STRING", "comment": "Unique transaction ID"},
                {"name": "customer_id", "type": "STRING", "comment": "Customer identifier"},
                {"name": "amount", "type": "DOUBLE", "comment": "Transaction amount"},
                {"name": "category", "type": "STRING", "comment": "Product category"},
                {"name": "transaction_date", "type": "TIMESTAMP", "comment": "Transaction timestamp"},
                {"name": "description", "type": "STRING", "comment": "Transaction description"},
                {"name": "status", "type": "STRING", "comment": "Transaction status"},
            ],
            "customers": [
                {"name": "customer_id", "type": "STRING", "comment": "Unique customer ID"},
                {"name": "first_name", "type": "STRING", "comment": "First name"},
                {"name": "last_name", "type": "STRING", "comment": "Last name"},
                {"name": "email", "type": "STRING", "comment": "Email address"},
                {"name": "phone", "type": "STRING", "comment": "Phone number"},
                {"name": "age", "type": "INT", "comment": "Customer age"},
                {"name": "income", "type": "DOUBLE", "comment": "Annual income"},
                {"name": "segment", "type": "STRING", "comment": "Customer segment"},
                {"name": "signup_date", "type": "DATE", "comment": "Signup date"},
                {"name": "notes", "type": "STRING", "comment": "Customer notes"},
            ],
        }
        return schemas.get(table, [])

    def add_mock_data(self, table: str, df: pd.DataFrame):
        """Add mock data for a table."""
        self._mock_data[table] = df

    def read_table(
        self,
        table: str,
        database: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        where: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return mock data or generate sample data."""
        if table in self._mock_data:
            df = self._mock_data[table]
        else:
            df = self._generate_sample_data(table)

        if columns:
            df = df[[c for c in columns if c in df.columns]]

        if limit:
            df = df.head(limit)

        return df

    def _generate_sample_data(self, table: str) -> pd.DataFrame:
        """Generate sample data for a table."""
        import numpy as np
        from faker import Faker

        fake = Faker()
        n_rows = 1000

        if table == "transactions":
            return pd.DataFrame(
                {
                    "transaction_id": [fake.uuid4() for _ in range(n_rows)],
                    "customer_id": [f"CUST_{i:05d}" for i in np.random.randint(1, 500, n_rows)],
                    "amount": np.random.exponential(100, n_rows).round(2),
                    "category": np.random.choice(
                        ["Electronics", "Clothing", "Food", "Home", "Sports"], n_rows
                    ),
                    "transaction_date": pd.date_range("2023-01-01", periods=n_rows, freq="h"),
                    "description": [fake.sentence(nb_words=6) for _ in range(n_rows)],
                    "status": np.random.choice(
                        ["completed", "pending", "cancelled"], n_rows, p=[0.85, 0.1, 0.05]
                    ),
                }
            )

        elif table == "customers":
            return pd.DataFrame(
                {
                    "customer_id": [f"CUST_{i:05d}" for i in range(1, n_rows + 1)],
                    "first_name": [fake.first_name() for _ in range(n_rows)],
                    "last_name": [fake.last_name() for _ in range(n_rows)],
                    "email": [fake.email() for _ in range(n_rows)],
                    "phone": [fake.phone_number() for _ in range(n_rows)],
                    "age": np.random.randint(18, 80, n_rows),
                    "income": np.random.normal(60000, 25000, n_rows).clip(20000).round(2),
                    "segment": np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n_rows),
                    "signup_date": pd.date_range("2020-01-01", periods=n_rows, freq="D")[:n_rows],
                    "notes": [fake.paragraph(nb_sentences=2) for _ in range(n_rows)],
                }
            )

        # Default sample data
        return pd.DataFrame(
            {
                "id": range(1, n_rows + 1),
                "value": np.random.randn(n_rows),
                "category": np.random.choice(["A", "B", "C"], n_rows),
                "text": [fake.sentence() for _ in range(n_rows)],
            }
        )

    def get_table_stats(self, table: str, database: Optional[str] = None) -> Dict[str, Any]:
        """Get mock table statistics."""
        schema = self.get_table_schema(table, database)
        return {
            "database": database or "default",
            "table": table,
            "row_count": 1000,
            "column_count": len(schema),
            "columns": schema,
        }
