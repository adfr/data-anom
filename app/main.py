"""
Synthetic Data Generator - Main Streamlit Application

A showcase UI for generating synthetic data from CDP Iceberg datasets.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.ui_components import (
    display_column_config,
    display_correlation_heatmap,
    display_data_preview,
    display_profile_summary,
    display_quality_report,
)
from app.services.cdp_connector import CDPConnector, MockCDPConnector
from app.services.data_profiler import DataProfiler
from app.services.synthetic_generator import SDVSyntheticGenerator, SyntheticDataGenerator

# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "source_data" not in st.session_state:
        st.session_state.source_data = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "column_config" not in st.session_state:
        st.session_state.column_config = None
    if "synthetic_data" not in st.session_state:
        st.session_state.synthetic_data = None
    if "connector" not in st.session_state:
        st.session_state.connector = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = None


def render_sidebar():
    """Render the sidebar with data source selection."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/data-protection.png",
            width=80,
        )
        st.title("Data Source")

        data_source = st.radio(
            "Select data source",
            ["CDP Iceberg (Demo)", "Upload CSV", "Sample Dataset"],
            key="data_source_radio",
        )

        st.divider()

        if data_source == "CDP Iceberg (Demo)":
            render_cdp_connection()
        elif data_source == "Upload CSV":
            render_csv_upload()
        else:
            render_sample_dataset()

        st.divider()

        # Generation settings
        st.subheader("Generation Settings")

        st.session_state.n_rows = st.number_input(
            "Number of rows to generate",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
        )

        st.session_state.generator_type = st.selectbox(
            "Generator Type",
            ["Basic (Fast)", "SDV Gaussian Copula", "SDV CTGAN"],
            help="Basic is faster, SDV provides better statistical preservation",
        )

        st.session_state.random_seed = st.number_input(
            "Random Seed (0 for random)",
            min_value=0,
            max_value=999999,
            value=42,
        )

        return data_source


def render_cdp_connection():
    """Render CDP connection settings."""
    st.subheader("CDP Connection")

    # Demo mode notice
    st.info("Demo mode: Using mock CDP connector with sample data")

    use_demo = st.checkbox("Use Demo Mode", value=True)

    if use_demo:
        if st.button("Connect (Demo)"):
            connector = MockCDPConnector()
            connector.connect()
            st.session_state.connector = connector
            st.success("Connected to Demo CDP")
    else:
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=21050)
        database = st.text_input("Database", value="default")

        if st.button("Connect"):
            st.warning("Real CDP connection requires proper credentials")

    # Table selection
    if st.session_state.connector and st.session_state.connector.is_connected():
        st.subheader("Select Table")

        databases = st.session_state.connector.list_databases()
        selected_db = st.selectbox("Database", databases)

        if selected_db:
            tables = st.session_state.connector.list_tables(selected_db)
            selected_table = st.selectbox("Table", tables)

            if st.button("Load Table"):
                with st.spinner("Loading data..."):
                    df = st.session_state.connector.read_table(
                        selected_table, selected_db, limit=10000
                    )
                    st.session_state.source_data = df
                    st.session_state.profile = None
                    st.session_state.column_config = None
                    st.session_state.synthetic_data = None
                    st.success(f"Loaded {len(df)} rows from {selected_db}.{selected_table}")


def render_csv_upload():
    """Render CSV upload section."""
    st.subheader("Upload CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file to generate synthetic data from",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.source_data = df
            st.session_state.profile = None
            st.session_state.column_config = None
            st.session_state.synthetic_data = None
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")


def render_sample_dataset():
    """Render sample dataset selection."""
    st.subheader("Sample Datasets")

    sample_options = {
        "Customer Data": "customers",
        "Transaction Data": "transactions",
        "Mixed Types Demo": "mixed",
    }

    selected_sample = st.selectbox("Select sample", list(sample_options.keys()))

    if st.button("Load Sample"):
        df = generate_sample_dataset(sample_options[selected_sample])
        st.session_state.source_data = df
        st.session_state.profile = None
        st.session_state.column_config = None
        st.session_state.synthetic_data = None
        st.success(f"Loaded sample dataset: {len(df)} rows")


def generate_sample_dataset(dataset_type: str) -> pd.DataFrame:
    """Generate a sample dataset for demo purposes."""
    import numpy as np
    from faker import Faker

    fake = Faker()
    n = 500

    if dataset_type == "customers":
        return pd.DataFrame(
            {
                "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],
                "first_name": [fake.first_name() for _ in range(n)],
                "last_name": [fake.last_name() for _ in range(n)],
                "email": [fake.email() for _ in range(n)],
                "phone": [fake.phone_number() for _ in range(n)],
                "age": np.random.randint(18, 80, n),
                "income": np.random.normal(60000, 25000, n).clip(20000).round(2),
                "segment": np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n),
                "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "notes": [fake.paragraph(nb_sentences=2) for _ in range(n)],
            }
        )

    elif dataset_type == "transactions":
        return pd.DataFrame(
            {
                "transaction_id": [fake.uuid4() for _ in range(n)],
                "customer_id": [f"CUST_{i:05d}" for i in np.random.randint(1, 200, n)],
                "amount": np.random.exponential(100, n).round(2),
                "category": np.random.choice(
                    ["Electronics", "Clothing", "Food", "Home", "Sports"], n
                ),
                "transaction_date": pd.date_range("2023-01-01", periods=n, freq="h"),
                "description": [fake.sentence(nb_words=6) for _ in range(n)],
                "status": np.random.choice(
                    ["completed", "pending", "cancelled"], n, p=[0.85, 0.1, 0.05]
                ),
                "is_fraud": np.random.choice([True, False], n, p=[0.02, 0.98]),
            }
        )

    else:  # mixed
        return pd.DataFrame(
            {
                "id": range(1, n + 1),
                "name": [fake.name() for _ in range(n)],
                "email": [fake.email() for _ in range(n)],
                "age": np.random.randint(18, 80, n),
                "salary": np.random.normal(50000, 15000, n).round(2),
                "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n),
                "is_active": np.random.choice([True, False], n, p=[0.9, 0.1]),
                "join_date": pd.date_range("2015-01-01", periods=n, freq="D"),
                "bio": [fake.paragraph() for _ in range(n)],
                "rating": np.random.uniform(1, 5, n).round(1),
            }
        )


def render_main_content():
    """Render the main content area."""
    st.markdown('<p class="main-header">Synthetic Data Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate privacy-safe synthetic data that preserves statistical properties</p>',
        unsafe_allow_html=True,
    )

    # Check if data is loaded
    if st.session_state.source_data is None:
        st.info("Please select a data source from the sidebar to get started.")

        # Show features
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Data Profiling")
            st.write(
                "Automatically detect column types and analyze distributions for optimal synthetic data generation."
            )

        with col2:
            st.markdown("### 🔄 Multiple Generators")
            st.write(
                "Choose from basic statistical generators or advanced SDV models like CTGAN for better preservation."
            )

        with col3:
            st.markdown("### 📈 Quality Reports")
            st.write(
                "Compare original and synthetic data distributions to ensure statistical properties are preserved."
            )

        return

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Data Preview", "🔍 Profile & Configure", "⚙️ Generate", "📊 Results"]
    )

    with tab1:
        render_data_preview_tab()

    with tab2:
        render_profile_tab()

    with tab3:
        render_generate_tab()

    with tab4:
        render_results_tab()


def render_data_preview_tab():
    """Render the data preview tab."""
    if st.session_state.source_data is not None:
        display_data_preview(st.session_state.source_data, "Source Data Preview")

        # Show column info
        st.subheader("Column Information")

        col_info = []
        for col in st.session_state.source_data.columns:
            series = st.session_state.source_data[col]
            col_info.append(
                {
                    "Column": col,
                    "Type": str(series.dtype),
                    "Non-Null": series.notna().sum(),
                    "Null %": f"{(series.isna().sum() / len(series) * 100):.1f}%",
                    "Unique": series.nunique(),
                    "Sample": str(series.dropna().iloc[0])[:50] if series.notna().any() else "N/A",
                }
            )

        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


def render_profile_tab():
    """Render the profile and configure tab."""
    if st.session_state.source_data is None:
        st.info("Load data first to view profile.")
        return

    # Profile button
    if st.button("Profile Data", type="primary"):
        with st.spinner("Profiling dataset..."):
            profiler = DataProfiler(st.session_state.source_data)
            st.session_state.profile = profiler.profile()
            st.session_state.column_config = profiler.get_column_config()

    # Display profile if available
    if st.session_state.profile:
        display_profile_summary(st.session_state.profile)

        # Correlation heatmap
        display_correlation_heatmap(st.session_state.source_data, "Column Correlations")

        # Column configuration
        st.divider()

        if st.session_state.column_config:
            updated_config = display_column_config(st.session_state.column_config)
            st.session_state.column_config = updated_config


def render_generate_tab():
    """Render the generate synthetic data tab."""
    if st.session_state.source_data is None:
        st.info("Load data first to generate synthetic data.")
        return

    st.subheader("Generate Synthetic Data")

    # Settings summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Source Rows", f"{len(st.session_state.source_data):,}")

    with col2:
        st.metric("Target Rows", f"{st.session_state.get('n_rows', 1000):,}")

    with col3:
        st.metric("Generator", st.session_state.get("generator_type", "Basic"))

    st.divider()

    # Generate button
    if st.button("Generate Synthetic Data", type="primary", use_container_width=True):
        generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic data based on current settings."""
    with st.spinner("Generating synthetic data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Get settings
            n_rows = st.session_state.get("n_rows", 1000)
            generator_type = st.session_state.get("generator_type", "Basic (Fast)")
            seed = st.session_state.get("random_seed", 42)
            seed = seed if seed > 0 else None

            # Get column config
            column_config = st.session_state.column_config or {}

            status_text.text("Initializing generator...")
            progress_bar.progress(10)

            if "SDV" in generator_type:
                # Use SDV generator
                model_type = "gaussian_copula" if "Gaussian" in generator_type else "ctgan"

                status_text.text(f"Fitting SDV {model_type} model...")
                progress_bar.progress(30)

                generator = SDVSyntheticGenerator(
                    st.session_state.source_data,
                    model_type=model_type,
                    random_seed=seed,
                )
                generator.fit()

                status_text.text("Generating synthetic samples...")
                progress_bar.progress(70)

                synthetic_df = generator.generate(n_rows)

            else:
                # Use basic generator
                status_text.text("Fitting basic generator...")
                progress_bar.progress(30)

                generator = SyntheticDataGenerator(
                    st.session_state.source_data,
                    column_config=column_config,
                    random_seed=seed,
                )
                generator.fit()

                status_text.text("Generating synthetic samples...")
                progress_bar.progress(70)

                synthetic_df = generator.generate(n_rows)

            progress_bar.progress(100)
            status_text.text("Generation complete!")

            st.session_state.synthetic_data = synthetic_df

            st.success(f"Successfully generated {len(synthetic_df)} synthetic rows!")

        except Exception as e:
            st.error(f"Error generating synthetic data: {e}")
            import traceback

            st.code(traceback.format_exc())


def render_results_tab():
    """Render the results tab."""
    if st.session_state.synthetic_data is None:
        st.info("Generate synthetic data first to view results.")
        return

    # Preview synthetic data
    display_data_preview(st.session_state.synthetic_data, "Synthetic Data Preview")

    st.divider()

    # Quality comparison
    display_quality_report(st.session_state.source_data, st.session_state.synthetic_data)

    st.divider()

    # Download options
    st.subheader("Download Synthetic Data")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = st.session_state.synthetic_data.to_csv(index=False)
        st.download_button(
            "Download as CSV",
            csv_data,
            file_name="synthetic_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        # Parquet download
        import io

        parquet_buffer = io.BytesIO()
        st.session_state.synthetic_data.to_parquet(parquet_buffer, index=False)
        parquet_data = parquet_buffer.getvalue()

        st.download_button(
            "Download as Parquet",
            parquet_data,
            file_name="synthetic_data.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )


def main():
    """Main application entry point."""
    init_session_state()
    data_source = render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
