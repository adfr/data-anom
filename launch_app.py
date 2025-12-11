#!/usr/bin/env python3
"""
Launcher script for CML Applications.

This script is used to launch the Streamlit app in CML environment.
"""

import os
import subprocess
import sys


def install_dependencies():
    """Install required dependencies if not already installed."""
    print("Checking and installing dependencies...")

    # Try multiple locations for requirements.txt
    possible_paths = [
        "requirements.txt",  # Current working directory
        "/home/cdsw/requirements.txt",  # CML default project path
    ]

    # Try to get script directory if available (for direct script execution)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.insert(0, os.path.join(script_dir, "requirements.txt"))
    except NameError:
        pass  # __file__ not defined in notebook context

    requirements_file = None
    for path in possible_paths:
        if os.path.exists(path):
            requirements_file = path
            break

    if requirements_file:
        print(f"Installing from {requirements_file}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file],
            check=True
        )
        print("Dependencies installed successfully.")
    else:
        # Fallback: install core dependencies directly
        print("requirements.txt not found, installing core dependencies...")
        core_deps = [
            "streamlit>=1.28.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "pyarrow>=14.0.0",
            "plotly>=5.18.0",
            "faker>=20.0.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.11.0",
            "pydantic>=2.5.0",
            "pydantic-settings>=2.1.0",
            "loguru>=0.7.0",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + core_deps,
            check=True
        )
        print("Core dependencies installed.")


def main():
    """Launch the Streamlit application."""
    # Install dependencies first
    install_dependencies()

    # Set environment variables for CML
    os.environ.setdefault("PYTHONPATH", "/home/cdsw")

    # Get port from environment or use default
    port = os.environ.get("CDSW_APP_PORT", "8080")

    # Build the streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/main.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    print(f"Launching Streamlit on port {port}...")
    print(f"Command: {' '.join(cmd)}")

    # Execute streamlit
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
