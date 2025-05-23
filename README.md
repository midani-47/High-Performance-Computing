# MPI-Based Fraud Detection Service

A high-performance distributed machine learning prediction service for fraud detection, built using MPI (Message Passing Interface). This service reads transaction data from a message queue, distributes prediction tasks across multiple processors/nodes, and sends prediction results back to a results queue.

## Overview

This service implements a distributed approach to ML prediction using OpenMPI to scale processing across multiple processors/nodes. The service:

1. Reads the pre-trained Random Forest model for fraud detection
2. Pulls transaction messages from a queue service (implemented in Assignment 3)
3. Distributes these transactions to worker nodes for parallel processing
4. Gathers prediction results from workers
5. Pushes prediction results to a results queue

## Features

- **Parallel Processing**: Utilizes MPI to distribute workload across multiple processors
- **Scalable Architecture**: Supports configurable number of worker processes
- **Queue Integration**: Seamlessly works with the message queue service from Assignment 3
- **Docker Support**: Includes containerization with Docker and multi-node setup with Docker Compose
- **Fault Tolerance**: Handles errors gracefully and continues processing available transactions
- **Configurable**: All parameters are configurable through environment variables or .env file

## System Requirements

- Python 3.8 or higher
- OpenMPI 4.0 or higher
- Docker and Docker Compose (for containerized deployment)
- Dependencies listed in requirements.txt

## Project Structure

```
/
├── prediction_service.py     # Main MPI-based prediction service
├── web_ui.py                # Web interface for testing
├── test_service.py          # Command-line test script
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── start_service.sh          # Shell script to start the service
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Multi-node Docker configuration
├── README.md                 # This file
└── DOCUMENTATION.md          # Detailed technical documentation
```

## Quick Setup (Docker-based)

The entire system can be started with a single Docker Compose command:

```bash
# Start everything with Docker Compose
docker-compose up -d

# To check logs
docker-compose logs -f

# To check logs for a specific service
docker-compose logs -f prediction_service
```

This will automatically:

1. Start the queue service on port 7501 (to avoid conflicts with existing services)
2. Start the prediction service
3. Launch a web UI for testing at http://localhost:7600

Once started, open http://localhost:7600 in your browser to access the web UI for testing.

### Alternative: Manual Setup

If you prefer to run components individually without Docker:

```bash
# Install OpenMPI
brew install open-mpi  # macOS
# or
sudo apt-get install openmpi-bin libopenmpi-dev  # Ubuntu

# Install Python dependencies
pip install -r requirements.txt

# Start the prediction service with 6 processes (1 master + 5 workers)
mpirun -n 6 python prediction_service.py
```

## Configuration

The service can be configured through environment variables or the `.env` file:

- `NUM_PROCESSORS`: Number of worker processors (default: 5)
- `QUEUE_SERVICE_URL`: URL of the queue service (default: http://localhost:7500)
- `TRANSACTION_QUEUE`: Name of the transaction queue (default: transactions)
- `RESULTS_QUEUE`: Name of the results queue (default: predictions)
- `MODEL_PATH`: Path to the pre-trained model (default: ./mpi/fraud_rf_model.pkl)
- `AUTH_USERNAME`: Username for queue service authentication (default: agent)
- `AUTH_PASSWORD`: Password for queue service authentication (default: agent_password)

## Usage

### One-Command Docker Approach

1. Run `docker-compose up -d` to start everything
2. Open http://localhost:7600 in your browser to access the web UI
3. Use the web UI to:
   - Set up the transaction and prediction queues (if they don't exist)
   - Push sample transactions to the transaction queue
   - Monitor prediction results as they are processed

### Understanding the Workflow

1. **Port 7501**: This is where the queue service runs. It handles the transaction and prediction queues.
2. **Port 7600**: This is where the web UI runs. It provides a user-friendly interface to interact with the queue service.
3. **Prediction Service**: This service reads transactions from the queue, processes them using simulated parallel processing, and sends results back to the prediction queue.

### Sample Transaction Format

The prediction service expects transactions in this format:

```json
{
  "transaction_id": "unique-id-123",
  "customer_id": "CUST_1234",
  "amount": 150.75,
  "vendor_id": "VENDOR_456"
}
```

The web UI will automatically generate sample transactions in the correct format.

### Web UI for Testing

The web-based user interface is automatically started as part of the Docker setup at http://localhost:7600. It provides:

- Status monitoring of the queue service and queues
- Queue creation functionality
- Sample transaction generation and submission
- Prediction result viewing

This UI makes it easy to test the entire system without needing to use the command line for testing.

See [DOCUMENTATION.md](DOCUMENTATION.md) for more detailed information about the implementation and usage of this service.



## Authors
- Nevin Joseph
- Abed Midani
