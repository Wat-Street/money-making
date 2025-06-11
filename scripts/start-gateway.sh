#!/bin/bash

echo "Starting WatStreet API Gateway..."

# Navigate to gateway directory
cd "$(dirname "$0")/../gateway"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the gateway
echo "Starting gateway on port 5000..."
python app.py 