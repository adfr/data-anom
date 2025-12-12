"""
Flask Backend API for Synthetic Data Generator.

Provides REST API endpoints for data loading, profiling, and synthetic data generation.
"""

import io
import json
import os
import sys
import traceback
from functools import wraps

from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np

from app.services.cdp_connector import MockCDPConnector
from app.services.data_profiler import DataProfiler
from app.services.synthetic_generator import SyntheticDataGenerator
from app.services.tokenizer import FormatPreservingTokenizer, detect_sensitive_columns
from app.services.llm_detector import LLMColumnDetector, is_llm_available

# Try to import CML connector
try:
    from app.services.cml_connector import CMLDataLakeConnector
    CML_AVAILABLE = True
except ImportError:
    CML_AVAILABLE = False

# Try to import SDV
try:
    from app.services.synthetic_generator import SDVSyntheticGenerator
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


# Get the app directory for templates
APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(APP_DIR, 'templates'))
CORS(app)

# Global state (in production, use Redis or similar)
state = {
    "connector": None,
    "connector_type": None,
    "source_data": None,
    "source_table": None,
    "profile": None,
    "column_config": None,
    "synthetic_data": None,
    "tokenizer": None,
    "tokenized_data": None,
    "tokenizer_key": None,
}


def is_cml_environment():
    """Check if running in CML environment."""
    return os.environ.get("CDSW_PROJECT_URL") is not None or os.environ.get("CDSW_ENGINE_ID") is not None


def handle_errors(f):
    """Decorator to handle errors in API endpoints."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    return wrapper


# ============ Frontend Route ============

@app.route("/")
def index():
    """Serve the React frontend."""
    return render_template("index.html")


# ============ Health & Info Endpoints ============

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "cml_environment": is_cml_environment()})


@app.route("/api/info", methods=["GET"])
def info():
    """Get application info and capabilities."""
    return jsonify({
        "version": "0.1.0",
        "cml_environment": is_cml_environment(),
        "cml_connector_available": CML_AVAILABLE,
        "sdv_available": SDV_AVAILABLE,
        "llm_available": is_llm_available(),
        "generator_types": ["basic"] + (["gaussian_copula", "ctgan"] if SDV_AVAILABLE else []),
        "state": {
            "connected": state["connector"] is not None,
            "connector_type": state["connector_type"],
            "data_loaded": state["source_data"] is not None,
            "data_rows": len(state["source_data"]) if state["source_data"] is not None else 0,
            "profiled": state["profile"] is not None,
            "synthetic_generated": state["synthetic_data"] is not None,
            "tokenized": state["tokenized_data"] is not None,
        }
    })


# ============ Connection Endpoints ============

@app.route("/api/connections", methods=["GET"])
@handle_errors
def list_connections():
    """List available CML Data Connections with their types."""
    if not CML_AVAILABLE:
        return jsonify({"connections": [], "error": "CML not available"})

    try:
        connector = CMLDataLakeConnector()
        connections = connector.get_available_connections()
        return jsonify({
            "connections": connections,
            "default_connection": os.environ.get("CML_CONNECTION_NAME"),
        })
    except Exception as e:
        return jsonify({"connections": [], "error": str(e)})


@app.route("/api/connect/demo", methods=["POST"])
@handle_errors
def connect_demo():
    """Connect using demo/mock connector."""
    connector = MockCDPConnector()
    connector.connect()
    state["connector"] = connector
    state["connector_type"] = "demo"
    return jsonify({"status": "connected", "type": "demo"})


@app.route("/api/connect/cml", methods=["POST"])
@handle_errors
def connect_cml():
    """Connect to Cloudera using CML Data Connections or Spark."""
    if not CML_AVAILABLE:
        return jsonify({"error": "CML connector not available. Make sure you're running in Cloudera ML."}), 400

    data = request.get_json() or {}
    # Use provided connection name, env variable, or auto-detect
    connection_name = data.get("connection_name") or os.environ.get("CML_CONNECTION_NAME")

    try:
        # Use get_connector() which checks CML_CONNECTION_TYPE and returns appropriate connector
        from app.services.cml_connector import get_connector
        connector = get_connector(connection_name=connection_name)
        state["connector"] = connector
        state["connector_type"] = "cml"

        # Get connection info
        conn_info = connector.get_connection_info()

        return jsonify({
            "status": "connected",
            "type": "cml",
            "connection_name": conn_info.get("name", "spark"),
            "connection_type": conn_info.get("type"),
            "connection_type_label": conn_info.get("type_label"),
            "available_connections": conn_info.get("available_connections", []),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/disconnect", methods=["POST"])
@handle_errors
def disconnect():
    """Disconnect from data source."""
    if state["connector"]:
        if hasattr(state["connector"], "close"):
            state["connector"].close()
        state["connector"] = None
        state["connector_type"] = None
    return jsonify({"status": "disconnected"})


# ============ Database/Table Endpoints ============

@app.route("/api/databases", methods=["GET"])
@handle_errors
def list_databases():
    """List available databases."""
    if not state["connector"]:
        return jsonify({"error": "Not connected"}), 400

    databases = state["connector"].list_databases()
    return jsonify({"databases": databases})


@app.route("/api/tables/<database>", methods=["GET"])
@handle_errors
def list_tables(database):
    """List tables in a database."""
    if not state["connector"]:
        return jsonify({"error": "Not connected"}), 400

    tables = state["connector"].list_tables(database)
    return jsonify({"database": database, "tables": tables})


@app.route("/api/table/<database>/<table>/schema", methods=["GET"])
@handle_errors
def get_table_schema(database, table):
    """Get table schema."""
    if not state["connector"]:
        return jsonify({"error": "Not connected"}), 400

    schema = state["connector"].get_table_schema(table, database)
    return jsonify({"database": database, "table": table, "schema": schema})


@app.route("/api/table/<database>/<table>/load", methods=["POST"])
@handle_errors
def load_table(database, table):
    """Load data from a table."""
    if not state["connector"]:
        return jsonify({"error": "Not connected"}), 400

    data = request.get_json() or {}
    limit = data.get("limit", 200)  # Default 200 for Spark performance

    # Load data
    if hasattr(state["connector"], "get_sample"):
        df = state["connector"].get_sample(table, database, n=limit)
    else:
        df = state["connector"].read_table(table, database, limit=limit)

    state["source_data"] = df
    state["source_table"] = f"{database}.{table}"
    state["profile"] = None
    state["column_config"] = None
    state["synthetic_data"] = None

    return jsonify({
        "status": "loaded",
        "table": f"{database}.{table}",
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    })


# ============ CSV Upload Endpoint ============

@app.route("/api/upload/csv", methods=["POST"])
@handle_errors
def upload_csv():
    """Upload a CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    df = pd.read_csv(file)

    state["source_data"] = df
    state["source_table"] = file.filename
    state["profile"] = None
    state["column_config"] = None
    state["synthetic_data"] = None

    return jsonify({
        "status": "loaded",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    })


# ============ Sample Data Endpoint ============

@app.route("/api/sample/<dataset_type>", methods=["POST"])
@handle_errors
def load_sample(dataset_type):
    """Load a sample dataset."""
    from faker import Faker

    fake = Faker()
    n = 500

    if dataset_type == "customers":
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],
            "first_name": [fake.first_name() for _ in range(n)],
            "last_name": [fake.last_name() for _ in range(n)],
            "email": [fake.email() for _ in range(n)],
            "phone": [fake.phone_number() for _ in range(n)],
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(60000, 25000, n).clip(20000).round(2),
            "segment": np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n),
        })
    elif dataset_type == "transactions":
        df = pd.DataFrame({
            "transaction_id": [fake.uuid4() for _ in range(n)],
            "customer_id": [f"CUST_{i:05d}" for i in np.random.randint(1, 200, n)],
            "amount": np.random.exponential(100, n).round(2),
            "category": np.random.choice(["Electronics", "Clothing", "Food", "Home", "Sports"], n),
            "description": [fake.sentence(nb_words=6) for _ in range(n)],
            "status": np.random.choice(["completed", "pending", "cancelled"], n, p=[0.85, 0.1, 0.05]),
        })
    else:  # mixed
        df = pd.DataFrame({
            "id": range(1, n + 1),
            "name": [fake.name() for _ in range(n)],
            "email": [fake.email() for _ in range(n)],
            "age": np.random.randint(18, 80, n),
            "salary": np.random.normal(50000, 15000, n).round(2),
            "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n),
            "is_active": np.random.choice([True, False], n, p=[0.9, 0.1]),
        })

    state["source_data"] = df
    state["source_table"] = f"sample_{dataset_type}"
    state["profile"] = None
    state["column_config"] = None
    state["synthetic_data"] = None

    return jsonify({
        "status": "loaded",
        "dataset": dataset_type,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    })


# ============ Profiling Endpoints ============

@app.route("/api/profile", methods=["POST"])
@handle_errors
def profile_data():
    """Profile the loaded data."""
    if state["source_data"] is None:
        return jsonify({"error": "No data loaded"}), 400

    profiler = DataProfiler(state["source_data"])
    profile = profiler.profile()
    column_config = profiler.get_column_config()

    state["profile"] = profile
    state["column_config"] = column_config

    # Convert numpy/pandas types to Python types for JSON serialization
    def convert_types(obj):
        if obj is None:
            return None
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return [convert_types(x) for x in obj.tolist()]
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_types(i) for i in obj]
        if isinstance(obj, (str, int, float)):
            return obj
        # Fallback: convert to string
        try:
            return str(obj)
        except:
            return None

    return jsonify({
        "profile": convert_types(profile),
        "column_config": convert_types(column_config),
    })


@app.route("/api/column-config", methods=["GET"])
@handle_errors
def get_column_config():
    """Get current column configuration."""
    if state["column_config"] is None:
        return jsonify({"error": "Data not profiled yet"}), 400
    return jsonify({"column_config": state["column_config"]})


@app.route("/api/column-config", methods=["PUT"])
@handle_errors
def update_column_config():
    """Update column configuration."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No configuration provided"}), 400

    state["column_config"] = data
    return jsonify({"status": "updated", "column_config": state["column_config"]})


# ============ Generation Endpoints ============

@app.route("/api/generate", methods=["POST"])
@handle_errors
def generate_synthetic():
    """Generate synthetic data."""
    if state["source_data"] is None:
        return jsonify({"error": "No data loaded"}), 400

    data = request.get_json() or {}
    n_rows = data.get("n_rows", 1000)
    generator_type = data.get("generator_type", "basic")
    random_seed = data.get("random_seed")

    if random_seed == 0:
        random_seed = None

    column_config = state["column_config"] or {}

    if generator_type == "basic":
        generator = SyntheticDataGenerator(
            state["source_data"],
            column_config=column_config,
            random_seed=random_seed
        )
        generator.fit()
        synthetic_df = generator.generate(n_rows)
    elif generator_type in ["gaussian_copula", "ctgan"] and SDV_AVAILABLE:
        model_type = generator_type
        generator = SDVSyntheticGenerator(
            state["source_data"],
            model_type=model_type,
            random_seed=random_seed
        )
        generator.fit()
        synthetic_df = generator.generate(n_rows)
    else:
        return jsonify({"error": f"Unknown generator type: {generator_type}"}), 400

    state["synthetic_data"] = synthetic_df

    return jsonify({
        "status": "generated",
        "rows": len(synthetic_df),
        "columns": list(synthetic_df.columns),
        "preview": synthetic_df.head(10).to_dict(orient="records"),
    })


@app.route("/api/synthetic/preview", methods=["GET"])
@handle_errors
def preview_synthetic():
    """Preview synthetic data."""
    if state["synthetic_data"] is None:
        return jsonify({"error": "No synthetic data generated"}), 400

    n = request.args.get("n", 20, type=int)
    return jsonify({
        "rows": len(state["synthetic_data"]),
        "columns": list(state["synthetic_data"].columns),
        "preview": state["synthetic_data"].head(n).to_dict(orient="records"),
    })


@app.route("/api/synthetic/download/<format>", methods=["GET"])
@handle_errors
def download_synthetic(format):
    """Download synthetic data."""
    if state["synthetic_data"] is None:
        return jsonify({"error": "No synthetic data generated"}), 400

    if format == "csv":
        output = io.StringIO()
        state["synthetic_data"].to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="synthetic_data.csv"
        )
    elif format == "parquet":
        output = io.BytesIO()
        state["synthetic_data"].to_parquet(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="synthetic_data.parquet"
        )
    else:
        return jsonify({"error": f"Unknown format: {format}"}), 400


# ============ Comparison Endpoints ============

@app.route("/api/compare/stats", methods=["GET"])
@handle_errors
def compare_stats():
    """Compare statistics between original and synthetic data."""
    if state["source_data"] is None:
        return jsonify({"error": "No source data"}), 400
    if state["synthetic_data"] is None:
        return jsonify({"error": "No synthetic data"}), 400

    original = state["source_data"]
    synthetic = state["synthetic_data"]

    comparison = []
    for col in original.columns:
        if col not in synthetic.columns:
            continue

        row = {"column": col}

        if pd.api.types.is_numeric_dtype(original[col]) and pd.api.types.is_numeric_dtype(synthetic[col]):
            row.update({
                "orig_mean": float(original[col].mean()) if not pd.isna(original[col].mean()) else None,
                "synth_mean": float(synthetic[col].mean()) if not pd.isna(synthetic[col].mean()) else None,
                "orig_std": float(original[col].std()) if not pd.isna(original[col].std()) else None,
                "synth_std": float(synthetic[col].std()) if not pd.isna(synthetic[col].std()) else None,
            })
        else:
            row.update({
                "orig_unique": int(original[col].nunique()),
                "synth_unique": int(synthetic[col].nunique()),
            })

        comparison.append(row)

    return jsonify({"comparison": comparison})


@app.route("/api/compare/distribution/<column>", methods=["GET"])
@handle_errors
def compare_distribution(column):
    """Get distribution data for comparison charts."""
    if state["source_data"] is None:
        return jsonify({"error": "No source data"}), 400
    if state["synthetic_data"] is None:
        return jsonify({"error": "No synthetic data"}), 400

    original = state["source_data"]
    synthetic = state["synthetic_data"]

    if column not in original.columns or column not in synthetic.columns:
        return jsonify({"error": f"Column {column} not found"}), 400

    if pd.api.types.is_numeric_dtype(original[column]) and pd.api.types.is_numeric_dtype(synthetic[column]):
        # Histogram data
        orig_values = original[column].dropna().tolist()
        synth_values = synthetic[column].dropna().tolist()
        return jsonify({
            "type": "numeric",
            "original": orig_values[:1000],  # Limit for performance
            "synthetic": synth_values[:1000],
        })
    else:
        # Category counts
        orig_counts = original[column].value_counts().head(20).to_dict()
        synth_counts = synthetic[column].value_counts().head(20).to_dict()
        return jsonify({
            "type": "categorical",
            "original": {str(k): int(v) for k, v in orig_counts.items()},
            "synthetic": {str(k): int(v) for k, v in synth_counts.items()},
        })


# ============ Tokenization Endpoints ============

@app.route("/api/tokenize/detect", methods=["POST"])
@handle_errors
def detect_sensitive():
    """Auto-detect sensitive columns in the loaded data."""
    if state["source_data"] is None:
        return jsonify({"error": "No data loaded"}), 400

    data = request.get_json() or {}
    use_llm = data.get("use_llm", False)
    additional_context = data.get("context")

    detection_method = "rule_based"

    if use_llm:
        # Get API key from request or environment
        api_key = data.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            return jsonify({
                "error": "LLM detection requires an API key. Provide 'api_key' in request or set ANTHROPIC_API_KEY environment variable."
            }), 400

        try:
            detector = LLMColumnDetector(api_key=api_key)
            sensitive = detector.detect_sensitive_columns(
                state["source_data"],
                additional_context=additional_context
            )
            detection_method = "llm"
        except Exception as e:
            # Fall back to rule-based
            logger.warning(f"LLM detection failed, falling back to rule-based: {e}")
            sensitive = detect_sensitive_columns(state["source_data"])
            detection_method = "rule_based_fallback"
    else:
        sensitive = detect_sensitive_columns(state["source_data"])

    return jsonify({
        "sensitive_columns": sensitive,
        "all_columns": list(state["source_data"].columns),
        "detection_method": detection_method,
        "llm_available": is_llm_available(),
    })


@app.route("/api/tokenize/generate-key", methods=["POST"])
@handle_errors
def generate_tokenizer_key():
    """Generate a new encryption key for tokenization."""
    key = FormatPreservingTokenizer.generate_key()
    return jsonify({"key": key})


@app.route("/api/tokenize", methods=["POST"])
@handle_errors
def tokenize_data():
    """Tokenize sensitive columns in the data."""
    if state["source_data"] is None:
        return jsonify({"error": "No data loaded"}), 400

    data = request.get_json() or {}
    secret_key = data.get("secret_key")
    column_types = data.get("column_types", {})

    if not secret_key:
        return jsonify({"error": "Secret key is required"}), 400

    if not column_types:
        return jsonify({"error": "No columns specified for tokenization"}), 400

    # Create tokenizer
    try:
        tokenizer = FormatPreservingTokenizer(secret_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Tokenize data
    tokenized_df = tokenizer.tokenize_dataframe(
        state["source_data"],
        column_types=column_types
    )

    state["tokenizer"] = tokenizer
    state["tokenized_data"] = tokenized_df
    state["tokenizer_key"] = secret_key

    # Get sample of original vs tokenized for preview
    preview_comparison = []
    for col in column_types.keys():
        if col in state["source_data"].columns:
            for i in range(min(5, len(state["source_data"]))):
                orig_val = state["source_data"][col].iloc[i]
                tok_val = tokenized_df[col].iloc[i]
                preview_comparison.append({
                    "column": col,
                    "original": str(orig_val) if pd.notna(orig_val) else None,
                    "tokenized": str(tok_val) if pd.notna(tok_val) else None,
                })

    return jsonify({
        "status": "tokenized",
        "rows": len(tokenized_df),
        "columns_tokenized": list(column_types.keys()),
        "preview": tokenized_df.head(10).to_dict(orient="records"),
        "comparison": preview_comparison,
    })


@app.route("/api/tokenize/preview", methods=["GET"])
@handle_errors
def preview_tokenized():
    """Preview tokenized data."""
    if state["tokenized_data"] is None:
        return jsonify({"error": "No tokenized data available"}), 400

    n = request.args.get("n", 20, type=int)
    return jsonify({
        "rows": len(state["tokenized_data"]),
        "columns": list(state["tokenized_data"].columns),
        "preview": state["tokenized_data"].head(n).to_dict(orient="records"),
    })


@app.route("/api/tokenize/download/<format>", methods=["GET"])
@handle_errors
def download_tokenized(format):
    """Download tokenized data."""
    if state["tokenized_data"] is None:
        return jsonify({"error": "No tokenized data available"}), 400

    if format == "csv":
        output = io.StringIO()
        state["tokenized_data"].to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="tokenized_data.csv"
        )
    elif format == "parquet":
        output = io.BytesIO()
        state["tokenized_data"].to_parquet(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="tokenized_data.parquet"
        )
    else:
        return jsonify({"error": f"Unknown format: {format}"}), 400


@app.route("/api/tokenize/mapping", methods=["GET"])
@handle_errors
def get_token_mapping():
    """Get the tokenization mapping (for audit purposes)."""
    if state["tokenizer"] is None:
        return jsonify({"error": "No tokenizer active"}), 400

    mapping = state["tokenizer"].get_token_mapping()
    return jsonify({
        "mapping_count": len(mapping),
        "mapping": mapping,
    })


@app.route("/api/tokenize/use-as-source", methods=["POST"])
@handle_errors
def use_tokenized_as_source():
    """Use the tokenized data as the new source for synthetic generation."""
    if state["tokenized_data"] is None:
        return jsonify({"error": "No tokenized data available"}), 400

    state["source_data"] = state["tokenized_data"].copy()
    state["profile"] = None
    state["column_config"] = None
    state["synthetic_data"] = None

    return jsonify({
        "status": "updated",
        "rows": len(state["source_data"]),
        "columns": list(state["source_data"].columns),
        "message": "Tokenized data is now the source for synthetic generation",
    })


def create_app():
    """Create and configure the Flask app."""
    return app


if __name__ == "__main__":
    port = int(os.environ.get("CDSW_APP_PORT", 8080))
    host = "127.0.0.1" if os.environ.get("CDSW_APP_PORT") else "0.0.0.0"

    print(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port, debug=False)
