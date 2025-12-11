#!/usr/bin/env python3
"""
Launcher script for CML Applications.

This script launches the Flask + React app in CML environment.
"""

import os
import subprocess
import sys


def install_dependencies():
    """Install required dependencies if not already installed."""
    print("Checking and installing dependencies...")

    # Use lightweight requirements for CML
    possible_paths = [
        "requirements-cml.txt",
        "/home/cdsw/requirements-cml.txt",
    ]

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
        print("Installing core dependencies...")
        core_deps = [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "plotly>=5.18.0",
            "faker>=20.0.0",
            "scikit-learn>=1.3.0",
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
    """Launch the Flask application."""
    # Install dependencies first
    install_dependencies()

    # Set environment variables for CML
    os.environ.setdefault("PYTHONPATH", "/home/cdsw")

    print("Starting Synthetic Data Generator application...")

    # CML uses 127.0.0.1 and CDSW_APP_PORT or CDSW_READONLY_PORT
    HOST = '127.0.0.1'
    PORT = os.getenv('CDSW_APP_PORT', os.getenv('CDSW_READONLY_PORT', '8090'))

    print(f"Running on {HOST}:{PORT}")

    # Import and run Flask app
    from app.api import app
    app.run(host=HOST, port=int(PORT))


if __name__ == "__main__":
    main()
