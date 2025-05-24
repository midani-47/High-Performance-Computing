# MPI-Based Fraud Detection Service

A high-performance distributed machine learning prediction service for fraud detection, built using MPI (Message Passing Interface). This service reads transaction data from local JSON files, distributes prediction tasks across multiple processors/nodes, and sends prediction results back to a results file.

## Overview

This service implements a distributed approach to ML prediction using OpenMPI to scale processing across multiple processors/nodes. The service:

1. Reads the pre-trained Random Forest model for fraud detection
2. Reads transaction data from local JSON files (TQ1.json and TQ2.json)
3. Distributes these transactions to worker nodes for parallel processing
4. Gathers prediction results from workers
5. Writes prediction results to a results file (PQ1.json)

## Features

- **Parallel Processing**: Utilizes MPI to distribute workload across multiple processors
- **Scalable Architecture**: Supports configurable number of worker processes
- **File-Based Queues**: Works with local JSON files for transaction data and prediction results
- **Docker Support**: Includes containerization with Docker and multi-node setup with Docker Compose
- **Fault Tolerance**: Handles errors gracefully and continues processing available transactions
- **Configurable**: All parameters are configurable through environment variables or .env file

## System Requirements

- Python 3.9 or higher
- OpenMPI 4.0 or higher
- Docker and Docker Compose (for containerized deployment)

### Dependencies
- mpi4py (for MPI support)
- numpy, pandas, scikit-learn (for data processing and ML)
- flask (for web UI)
- python-dotenv (for environment variables)
- joblib (for model loading)

All dependencies are automatically installed in the Docker containers using conda environments to avoid memory issues during builds.

## Project Structure

```
/
├── prediction_service.py     # Main MPI-based prediction service
├── web_ui_file.py           # Web interface for testing with file-based queues
├── a3/
│   └── queue_service/
│       └── queue_data/       # Directory containing queue JSON files
│           ├── TQ1.json      # Transaction Queue 1
│           ├── TQ2.json      # Transaction Queue 2
│           └── PQ1.json      # Prediction Results Queue
├── mpi/
│   └── fraud_rf_model.pkl    # Pre-trained fraud detection model
├── .env                      # Environment configuration
├── Dockerfile                # Docker configuration for prediction service
├── Dockerfile.web_ui        # Docker configuration for web UI
├── docker-compose.yml        # Multi-node Docker configuration
└── README.md                 # This file
```

## Quick Setup (Docker-based)

Before starting the Docker containers, run the initialization script to set up the necessary directories and files:

```bash
# Run the initialization script to set up the environment
./init_setup.sh
```

This script will:
1. Create the necessary directories for queue data and the model
2. Generate the fraud detection model if it doesn't exist
3. Create empty queue files if they don't exist
4. Set proper permissions for the queue data directory

After initialization, the entire system can be started with a single Docker Compose command:

```bash
# For first-time setup (or after changes), use the --build flag
docker-compose up --build -d

# For subsequent runs
docker-compose up -d

# To check logs
docker-compose logs -f

# To check logs for a specific service
docker-compose logs -f mpi_master
```

This will automatically:

1. Start the MPI master node and 5 worker nodes
2. Launch a web UI for testing at http://localhost:7600

Once started, open http://localhost:7600 in your browser to access the web UI for testing.

### Alternative: Manual Setup

If you prefer to run components individually without Docker:

```bash
# Install OpenMPI
brew install open-mpi  # macOS
# or
sudo apt-get install openmpi-bin libopenmpi-dev  # Ubuntu

# Install Python dependencies
pip install mpi4py numpy pandas scikit-learn python-dotenv joblib

# Start the prediction service with 6 processes (1 master + 5 workers)
mpirun -n 6 python prediction_service.py

# Start the web UI in a separate terminal
python web_ui_file.py
```

## Configuration

The service can be configured through environment variables or the `.env` file:

- `NUM_PROCESSORS`: Number of processors including master (default: 6)
- `QUEUE_DATA_DIR`: Directory containing queue JSON files (default: ./a3/queue_service/queue_data)
- `TRANSACTION_QUEUE_FILE`: Name of the first transaction queue file (default: TQ1.json)
- `TRANSACTION_QUEUE_FILE2`: Name of the second transaction queue file (default: TQ2.json)
- `RESULTS_QUEUE_FILE`: Name of the prediction results queue file (default: PQ1.json)
- `MODEL_PATH`: Path to the pre-trained model (default: ./mpi/fraud_rf_model.pkl)

## Usage

### One-Command Docker Approach

1. Run `docker-compose up --build -d` to start everything (first time)
2. Open http://localhost:7600 in your browser to access the web UI
3. Use the web UI to:
   - View the status of the transaction and prediction files
   - Push sample transactions to the transaction files (TQ1.json or TQ2.json)
   - Monitor prediction results as they are processed and written to PQ1.json

### Understanding the Workflow

1. **File-Based Queues**: The system uses JSON files in the `a3/queue_service/queue_data` directory to store transactions and prediction results.
2. **Port 7600**: This is where the web UI runs. It provides a user-friendly interface to interact with the queue files.
3. **MPI Prediction Service**: This service reads transactions from the queue files, processes them using parallel processing with MPI, and writes results back to the prediction results file.

### Sample Transaction Format

The prediction service expects transactions in this format:

```json
{
  "transaction_id": "unique-id-123",
  "customer_id": "CUST_1234",
  "customer_name": "John Doe",
  "amount": 150.75,
  "vendor_id": "VENDOR_456",
  "date": "2025-05-24"
}
```

The web UI will automatically generate sample transactions in the correct format.

### Web UI for Testing

The web-based user interface is automatically started as part of the Docker setup at http://localhost:7600. It provides:

- Status monitoring of the transaction and prediction files
- Sample transaction generation and submission
- Prediction result viewing

This UI makes it easy to test the entire system without needing to use the command line for testing.



## Authors
- Nevin Joseph
- Abed Midani
