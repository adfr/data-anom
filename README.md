# Synthetic Data Generator

A showcase UI for generating synthetic data from CDP Iceberg datasets. This tool enables ML engineers to create privacy-safe training data that preserves the statistical properties of original datasets.

## Features

- **CDP/Iceberg Integration**: Connect to Cloudera CDP and load Iceberg tables
- **Automatic Data Profiling**: Detect column types (continuous, categorical, text, free text, datetime, etc.)
- **Multiple Generation Methods**:
  - Basic statistical generators (fast)
  - SDV Gaussian Copula (statistical preservation)
  - SDV CTGAN (deep learning-based)
- **Column Type Support**:
  - **Continuous**: Gaussian/KDE-based generation
  - **Categorical**: Frequency-based sampling
  - **Text**: Faker-based realistic text
  - **Free Text**: Markov chain generation
  - **DateTime**: Range-based generation
  - **IDs/Emails/Phones**: Specialized fake data
- **Quality Reports**: Compare distributions between original and synthetic data
- **Export Options**: Download as CSV or Parquet

## Installation

### Cloudera AI (CML) Deployment (Recommended)

This application is designed to run on Cloudera AI with direct access to the CDP Data Lake.

1. **Create a new CML Project**:
   - Go to your CML workspace
   - Click "New Project"
   - Select "Git" and enter the repository URL

2. **Build the Project**:
   - The `cdsw-build.sh` script will automatically install dependencies

3. **Create an Application**:
   - Go to "Applications" in your project
   - Click "New Application"
   - Settings:
     - **Name**: Synthetic Data Generator
     - **Script**: `launch_app.py`
     - **Engine**: Python 3.10 with Spark addon
     - **Resource Profile**: 2 vCPU, 8GB RAM (minimum)
   - Click "Create Application"

4. **Access the Application**:
   - Once running, click the application URL
   - The UI will have direct access to your CDP Data Lake tables

### Local Development

For local development and testing:

1. Clone the repository:
```bash
git clone <repository-url>
cd data-anom
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For Spark/Iceberg support locally:
```bash
pip install pyspark>=3.4.0
```

## Usage

### Running the UI

**In CML:**
The application runs automatically when you start the CML Application.

**Locally:**
```bash
python launch_app.py
```

The application will open in your browser at `http://localhost:8090`.

### Quick Start

1. **Select Data Source**:
   - Use "Sample Dataset" for quick demo
   - Upload a CSV file
   - Connect to CDP Iceberg (demo mode available)

2. **Profile Data**:
   - Click "Profile Data" to analyze column types
   - Review detected types and recommendations
   - Adjust column configuration if needed

3. **Generate Synthetic Data**:
   - Set the number of rows to generate
   - Choose generator type (Basic or SDV)
   - Click "Generate Synthetic Data"

4. **Review & Download**:
   - Compare original and synthetic distributions
   - Download as CSV or Parquet

### CDP Connection

For real CDP connections, configure the following environment variables:

```bash
CDP_HOST=your-cdp-host.example.com
CDP_PORT=21050
CDP_DATABASE=your_database
CDP_USER=your_username
CDP_PASSWORD=your_password
CDP_USE_SSL=true
```

## Project Structure

```
data-anom/
├── app/
│   ├── __init__.py
│   ├── api.py                     # Flask REST API backend
│   ├── templates/
│   │   └── index.html             # React frontend UI
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cdp_connector.py       # CDP/Impala connection
│   │   ├── cml_connector.py       # CML native Spark/Iceberg connector
│   │   ├── data_profiler.py       # Data profiling service
│   │   └── synthetic_generator.py # Synthetic data generation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
├── config/
│   ├── __init__.py
│   └── settings.py                # Application settings
├── cdsw-build.sh                  # CML build script
├── launch_app.py                  # CML application launcher
├── .project-metadata.yaml         # CML project metadata
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Generation Methods

### Basic Generator (Fast)
- **Continuous columns**: Gaussian distribution matching mean/std
- **Categorical columns**: Frequency-based sampling
- **Text columns**: Faker library for realistic fake data
- **Free text**: Simple Markov chain based on word transitions

### SDV Gaussian Copula
- Captures correlations between columns
- Better statistical preservation
- Suitable for most tabular data

### SDV CTGAN
- Deep learning-based generation
- Best for complex distributions
- Slower but highest quality

## Column Types

| Type | Description | Generation Method |
|------|-------------|-------------------|
| `continuous` | Numeric values with many unique values | Gaussian/KDE |
| `categorical` | Limited set of values | Frequency sampling |
| `text` | Short text strings | Faker text |
| `free_text` | Long form text/paragraphs | Markov chain |
| `datetime` | Date/time values | Range sampling |
| `boolean` | True/False values | Bernoulli |
| `id` | Unique identifiers | UUID/Sequential |
| `email` | Email addresses | Faker email |
| `phone` | Phone numbers | Faker phone |
| `name` | Person names | Faker name |
| `address` | Physical addresses | Faker address |

## API Usage

You can also use the generators programmatically:

```python
import pandas as pd
from app.services.synthetic_generator import SyntheticDataGenerator
from app.services.data_profiler import DataProfiler

# Load your data
df = pd.read_csv("your_data.csv")

# Profile the data
profiler = DataProfiler(df)
profile = profiler.profile()
column_config = profiler.get_column_config()

# Generate synthetic data
generator = SyntheticDataGenerator(
    source_df=df,
    column_config=column_config,
    random_seed=42
)
generator.fit()
synthetic_df = generator.generate(n_rows=1000)

# Save results
synthetic_df.to_csv("synthetic_data.csv", index=False)
```

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `flask` - Web framework
- `pandas` - Data manipulation
- `sdv` - Synthetic Data Vault
- `faker` - Fake data generation
- `impyla` - CDP/Impala connection
- `plotly` - Visualizations

## License

MIT License
