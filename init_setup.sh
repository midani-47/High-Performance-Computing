#!/bin/bash
# Initialization script to set up the environment before running Docker containers

# Create necessary directories
mkdir -p a3/queue_service/queue_data
mkdir -p mpi

# Check if model exists, if not create it
if [ ! -f mpi/fraud_rf_model.pkl ]; then
    echo "Creating fraud detection model..."
    python create_model.py
fi

# Check if queue files exist, if not create them
if [ ! -f a3/queue_service/queue_data/TQ1.json ]; then
    echo "Creating TQ1.json..."
    echo "[]" > a3/queue_service/queue_data/TQ1.json
fi

if [ ! -f a3/queue_service/queue_data/TQ2.json ]; then
    echo "Creating TQ2.json..."
    echo "[]" > a3/queue_service/queue_data/TQ2.json
fi

if [ ! -f a3/queue_service/queue_data/PQ1.json ]; then
    echo "Creating PQ1.json..."
    echo "[]" > a3/queue_service/queue_data/PQ1.json
fi

# Set proper permissions
chmod -R 777 a3/queue_service/queue_data

echo "Initialization complete. You can now run 'docker-compose up --build -d'"
