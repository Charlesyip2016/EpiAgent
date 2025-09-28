#!/bin/bash

# Environment setup script for GWAS-Enformer-EpiAgent Workflow
# This script sets up the conda environment and installs all necessary dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="gwas_enformer_epiagent"

echo "Setting up environment for GWAS-Enformer-EpiAgent Workflow..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda env create -f "$SCRIPT_DIR/environment.yml" --name $ENV_NAME

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate $ENV_NAME"
echo ""
echo "To verify the installation, run:"
echo "conda activate $ENV_NAME"
echo "python -c \"import tensorflow as tf, torch, pandas as pd, epiagent; print('All packages imported successfully')\""