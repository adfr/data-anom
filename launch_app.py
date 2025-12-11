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

    # Use lightweight requirements for CML (avoids heavy SDV package)
    # Prefer requirements-cml.txt which excludes memory-heavy packages
    possible_paths = [
        "requirements-cml.txt",  # Lightweight for CML
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
        # Fallback: install core dependencies directly (lightweight)
        print("Installing core dependencies...")
        core_deps = [
            "streamlit>=1.28.0",
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
    """Launch the Streamlit application."""
    # Install dependencies first
    install_dependencies()

    # Set environment variables for CML
    os.environ.setdefault("PYTHONPATH", "/home/cdsw")

    # Get port from environment or use default
    port = os.environ.get("CDSW_APP_PORT", "8080")

    # Build the streamlit command with CML-compatible settings
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/main.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableXsrfProtection", "false",  # Required for CML
        "--server.enableCORS", "false",  # Required for CML
        "--browser.gatherUsageStats", "false",
    ]

    print(f"Launching Streamlit on port {port}...")
    print(f"Command: {' '.join(cmd)}")

    # Execute streamlit
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
