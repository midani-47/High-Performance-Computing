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

The system is designed for fast setup and deployment. Follow these steps:

### Step 1: Run the initialization script

**On Windows:**
```cmd
init_setup.bat
```

**On macOS/Linux:**
```bash
./init_setup.sh
```

This script will:
1. Create the necessary directories for queue data and the model
2. Generate the fraud detection model if it doesn't exist
3. Create empty queue files if they don't exist
4. Set proper permissions for the queue data directories

### Step 2: Start the system with a single command

```
docker-compose up --build -d
```

This single command will:
1. Build and start the MPI prediction service with 1 master and 5 worker processes
2. Build and start the web UI service
3. Create the necessary Docker network

### Additional Commands

```
# If you need to restart just the MPI prediction service
docker-compose up --build -d mpi_service

# If you need to restart just the web UI
docker-compose up --build -d web_ui

# To view logs of all services
docker-compose logs -f

# To view logs of just the MPI prediction service
docker-compose logs -f mpi_service

# To stop all services
docker-compose down
```

The system consists of:

1. **Web UI** - Runs on http://localhost:7600 and provides an interface to:
   - View queue status
   - Push sample transactions
   - View prediction results

2. **MPI Prediction Service** - A single container that runs multiple MPI processes:
   - Runs 6 MPI processes (1 master + 5 workers) within a single container
   - Reads transactions from the queue files (TQ1.json and TQ2.json)
   - Processes transactions in parallel using MPI
   - Writes prediction results to the results file (PQ1.json)

Once started, open http://localhost:7600 in your browser to access the web UI for testing.

### Alternative: Manual Setup

If you prefer to run components individually without Docker:

#### Option 1: Simple Setup (No MPI Required)

```bash
# 1. Install Python dependencies
pip install numpy pandas scikit-learn python-dotenv joblib flask requests

# 2. Run the initialization script to set up directories and files
./init_setup.sh

# 3. Start the web UI in one terminal
python3 web_ui_file.py

# 4. Start the prediction service in another terminal
python3 prediction_service.py
```

#### Option 2: MPI-based Parallel Processing

```bash
# 1. Install OpenMPI
brew install open-mpi  # macOS
# or
sudo apt-get install openmpi-bin libopenmpi-dev  # Ubuntu

# 2. Install Python dependencies including MPI
pip install mpi4py numpy pandas scikit-learn python-dotenv joblib flask requests

# 3. Run the initialization script to set up directories and files
./init_setup.sh

# 4. Start the web UI in one terminal
python3 web_ui_file.py

# 5. Start the prediction service with MPI in another terminal
mpirun -n 6 python3 prediction_service.py
```

**Note**: The prediction service is designed to work both with and without MPI. When running manually, make sure to:
1. Start the web UI first
2. Use the web UI to push some sample transactions to the queue
3. Then start the prediction service to process those transactions

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
