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

## Quick Setup

### Local Setup

```bash
# Install OpenMPI (if not already installed)
# On macOS:
brew install open-mpi

# On Ubuntu:
sudo apt-get install openmpi-bin libopenmpi-dev

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the service with 6 processes (1 master + 5 workers)
./start_service.sh 6
```

### Using Docker Compose (Recommended)

```bash
# Start the service using Docker Compose
docker-compose up --build

# To run in background
docker-compose up -d

# To check logs
docker-compose logs -f
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

1. Ensure the queue service from Assignment 3 is running
2. Create transaction and prediction queues if they don't exist
3. Start the prediction service using one of the methods above
4. Push transaction messages to the transaction queue
5. The service will automatically process transactions and push predictions to the results queue

### Web UI for Testing

A web-based user interface is provided for easy testing of the prediction service:

```bash
# Start the web UI
python web_ui.py
```

Then open http://localhost:7600 in your browser. The web UI allows you to:

- Check the status of the queue service and queues
- Create the required queues if they don't exist
- Push sample transactions to the transaction queue
- View prediction results from the results queue

### Command-line Testing

You can also test the service using the included test script:

```bash
# Run the test script to create queues and push sample transactions
python test_service.py

# Check for results after the service has processed transactions
python test_service.py --check-only
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for more detailed information about the implementation and usage of this service.



## Authors
- Nevin Joseph
- Abed Midani
