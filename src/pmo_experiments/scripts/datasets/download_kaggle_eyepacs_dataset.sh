#!/bin/bash

# Get the folder of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Move to the kaggle eyepacs dataset directory
# <PROJECT_ROOT>/datasets/kaggle-eyepacs
# Move to project root
cd "$SCRIPT_DIR/../../.."

# Activate virtual environment if exists else assume environment is already set up
if [ -d "./.venv" ]; then
    echo "Activating virtual environment in .venv..."
    source ./.venv/bin/activate
else
    echo "No virtual environment found in .venv, assuming environment is already set up."
fi

# Create dataset directory if it doesn't exist
mkdir -p datasets/kaggle-eyepacs

# Change to dataset directory
cd datasets/kaggle-eyepacs

# Check if file already exists
if [ -f "diabetic-retinopathy-detection.zip" ]; then
    echo "Dataset already downloaded."
    exit 0
fi

# Check if kaggle.json file exists in the user's home directory
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    # Check if env KAGGLE_API_TOKEN is set
    if [ -z "$KAGGLE_API_TOKEN" ];
    then
        echo "Kaggle API token not found. Please set the KAGGLE_API_TOKEN environment variable or place kaggle.json in ~/.kaggle/"
        exit 1
    fi
fi

# Download the dataset using kaggle API
kaggle competitions download -c diabetic-retinopathy-detection
