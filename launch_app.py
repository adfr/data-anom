#!/usr/bin/env python3
"""
Launcher script for CML Applications.

This script is used to launch the Streamlit app in CML environment.
"""

import os
import subprocess
import sys


def main():
    """Launch the Streamlit application."""
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
