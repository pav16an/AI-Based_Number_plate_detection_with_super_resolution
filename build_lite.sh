#!/usr/bin/env bash
set -o errexit
pip install --upgrade pip
pip install flask gunicorn
mkdir -p uploads json
echo "Build completed"