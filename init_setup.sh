#!/bin/bash
# Initialization script to set up the environment before running Docker containers

echo "Setting up environment for MPI Prediction Service..."

# Create necessary directories
mkdir -p a3/queue_service/queue_data
mkdir -p mpi

# Generate model if it doesn't exist
if [ ! -f "mpi/fraud_rf_model.pkl" ]; then

# Write empty arrays to queue files
echo '[]' > a3/queue_service/queue_data/TQ1.json
echo '[]' > a3/queue_service/queue_data/TQ2.json
echo '[]' > a3/queue_service/queue_data/PQ1.json

# Ensure permissions are set correctly
chmod -R 777 a3/queue_service/queue_data
chmod -R 777 mpi

# Generate the model
python3 create_model.py

# Verify the model was created
if [ -f "mpi/fraud_rf_model.pkl" ]; then
    echo "Model created successfully at mpi/fraud_rf_model.pkl"
else
    echo "WARNING: Model creation failed. The prediction service may use a mock model."
fi

echo "Initialization complete. You can now start the prediction service."
echo "You can now run: docker-compose up --build -d"
