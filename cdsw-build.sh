#!/bin/bash
# CML Build Script for Synthetic Data Generator
# This script runs during the CML project build phase

set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build complete ==="
