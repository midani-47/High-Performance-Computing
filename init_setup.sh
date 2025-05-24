#!/bin/bash
# Initialization script to set up the environment before running Docker containers

echo "Setting up environment for MPI Prediction Service..."

# Create necessary directories
mkdir -p a3/queue_service/queue_data
mkdir -p mpi

# Generate model if it doesn't exist
if [ ! -f "mpi/fraud_rf_model.pkl" ]; then
    echo "Generating fraud detection model..."
    python3 create_model.py
fi

# Create empty queue files if they don't exist
if [ ! -f "a3/queue_service/queue_data/TQ1.json" ]; then
    echo "Creating empty TQ1.json..."
    echo "[]" > a3/queue_service/queue_data/TQ1.json
fi

if [ ! -f "a3/queue_service/queue_data/TQ2.json" ]; then
    echo "Creating empty TQ2.json..."
    echo "[]" > a3/queue_service/queue_data/TQ2.json
fi

if [ ! -f "a3/queue_service/queue_data/PQ1.json" ]; then
    echo "Creating empty PQ1.json..."
    echo "[]" > a3/queue_service/queue_data/PQ1.json
fi

# Set permissions for queue data directory
chmod -R 777 a3/queue_service/queue_data

echo "Environment setup complete!"
echo "You can now run: docker-compose up --build -d"
