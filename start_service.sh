#!/bin/bash
# Script to start the MPI-based prediction service

# Default number of processes
NUM_PROCESSES=${1:-6}  # Default: 5 workers + 1 master process

# Check if mpirun is installed
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun is not installed. Please install OpenMPI first."
    echo "On macOS: brew install open-mpi"
    echo "On Ubuntu: sudo apt-get install openmpi-bin libopenmpi-dev"
    exit 1
fi

# Check if the model file exists
MODEL_PATH="./mpi/fraud_rf_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file is in the correct location."
fi

# Run the MPI application
echo "Starting prediction service with $((NUM_PROCESSES-1)) worker processes..."
mpirun -n $NUM_PROCESSES python prediction_service.py
