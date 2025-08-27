#!/usr/bin/env bash
# Lightweight build script for Render deployment

set -o errexit

# Install Python dependencies
export SETUPTOOLS_USE_DISTUTILS=stdlib
pip install --upgrade pip
pip install setuptools wheel
pip install --no-cache-dir -r requirements_deploy.txt

# Create necessary directories
mkdir -p uploads
mkdir -p json

echo "Lightweight build completed successfully"