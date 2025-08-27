#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements_deploy.txt

# Create necessary directories
mkdir -p uploads
mkdir -p json
mkdir -p weights

echo "Build completed successfully"