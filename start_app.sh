#!/bin/bash
# Shell script to run the PyCaret ML App with Ngrok

echo "============================================================"
echo "PyCaret ML App - Easy Launcher"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "[2/3] Starting application..."
echo ""
echo "Choose how to run the app:"
echo "1. Run locally (only accessible on this computer)"
echo "2. Run with Ngrok (publicly accessible via internet)"
echo ""
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo ""
    echo "Starting locally at http://localhost:8501"
    echo "Press Ctrl+C to stop"
    echo ""
    streamlit run main.py
elif [ "$choice" == "2" ]; then
    echo ""
    echo "Starting with Ngrok..."
    python3 run_with_ngrok.py
else
    echo "Invalid choice. Please run the script again."
fi
