#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/images/uploaded
mkdir -p static/images/processed
mkdir -p static/images/results
mkdir -p output

# Set permissions
chmod -R 755 static
chmod -R 755 output 