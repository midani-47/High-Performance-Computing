# MPI-Based Fraud Detection Service

A high-performance distributed machine learning prediction service for fraud detection, built using MPI (Message Passing Interface). This service reads transaction data from queue files, distributes prediction tasks across multiple processors, and sends prediction results back to a results queue file.

## Overview

This service implements a distributed approach to ML prediction using OpenMPI to scale processing across multiple processors. The service:

1. Reads the pre-trained Random Forest model for fraud detection (`fraud_rf_model.pkl`)
2. Reads transaction data from queue files (TQ1.json and TQ2.json) that were implemented in Assignment 3
3. Distributes these transactions to worker processes for parallel processing
4. Gathers prediction results from workers
5. Writes prediction results to a results file (PQ1.json)

## Features

- **Parallel Processing**: Utilizes MPI to distribute workload across multiple processors
- **Scalable Architecture**: Supports configurable number of worker processes
- **File-Based Queues**: Works with local JSON files for transaction data and prediction results
- **Integration with Assignment 3**: Compatible with the queue service from Assignment 3
- **Web UI**: Provides a simple web interface for monitoring and testing
- **Fault Tolerance**: Handles errors gracefully and continues processing available transactions
- **Configurable**: Number of processors and other parameters are configurable through environment variables

## System Requirements

- Python 3.8 or higher
- OpenMPI 4.0 or higher
- Required Python packages: mpi4py, numpy, pandas, scikit-learn, flask

## Project Structure

```
/
├── simple_prediction_service.py  # Main MPI-based prediction service
├── simple_create_model.py        # Script to generate the ML model
├── web_ui.py                     # Web UI for monitoring and testing
├── test_mpi_service.ps1          # PowerShell test script
├── test_mpi_service.sh           # Bash test script
├── a3/
│   └── queue_service/
│       ├── app/                  # Assignment 3 queue service implementation
│       │   ├── models.py         # Data models for transactions and predictions
│       │   └── ...
│       └── queue_data/           # Directory containing queue JSON files
│           ├── TQ1.json          # Transaction Queue 1
│           ├── TQ2.json          # Transaction Queue 2
│           └── PQ1.json          # Prediction Results Queue
├── mpi/
│   └── fraud_rf_model.pkl        # Pre-trained fraud detection model
└── templates/
    └── index.html                # Web UI template
```

## Quick Setup

### Step 1: Install Dependencies

```bash
# Install OpenMPI
# On Ubuntu:
sudo apt-get install openmpi-bin libopenmpi-dev

# On macOS:
brew install open-mpi

# On Windows:
# Install Microsoft MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Run the Test Script

On Windows:
```powershell
.\test_mpi_service.ps1
```

On Linux/macOS:
```bash
chmod +x test_mpi_service.sh
./test_mpi_service.sh
```

This will:
1. Create necessary directories and queue files
2. Generate the model if needed
3. Add a test transaction to the queue
4. Start the web UI
5. Provide instructions for running the MPI service

### Step 3: Run the MPI Service

In a separate terminal, run:

```bash
# On Linux/macOS:
mpirun -n 5 python simple_prediction_service.py

# On Windows:
mpiexec -n 5 python simple_prediction_service.py
```

This will:
1. Start 5 MPI processes (1 master + 4 workers)
2. Process transactions from the queue files
3. Write results to the results file

### Step 4: Access the Web UI

Open your browser and navigate to:
```
http://localhost:7600
```

The web UI allows you to:
- Monitor queue status
- Push sample transactions
- View prediction results

## Configuration

The service can be configured through environment variables:

- `NUM_PROCESSORS`: Number of processors including master (default: 5)
- `QUEUE_DATA_DIR`: Directory containing queue JSON files (default: ./a3/queue_service/queue_data)
- `TRANSACTION_QUEUE_FILE`: Name of the first transaction queue file (default: TQ1.json)
- `TRANSACTION_QUEUE_FILE2`: Name of the second transaction queue file (default: TQ2.json)
- `RESULTS_QUEUE_FILE`: Name of the prediction results queue file (default: PQ1.json)
- `MODEL_PATH`: Path to the pre-trained model (default: ./mpi/fraud_rf_model.pkl)

## Transaction Format

The prediction service supports both simple transaction format and Assignment 3 message format:

### Simple Transaction Format
```json
{
  "transaction_id": "tx123",
  "amount": 1000.00,
  "transaction_count": 5,
  "customer_risk_score": 0.3,
  "vendor_risk_score": 0.2
}
```

### Assignment 3 Message Format
```json
{
  "content": {
    "transaction_id": "tx123",
    "customer_id": "cust456",
    "customer_name": "John Doe",
    "amount": 1000.00,
    "vendor_id": "vend789",
    "date": "2023-05-01T12:34:56",
    "transaction_count": 5,
    "customer_risk_score": 0.3,
    "vendor_risk_score": 0.2
  },
  "timestamp": "2023-05-01T12:34:56",
  "message_type": "transaction",
  "id": "msg123"
}
```

## Authors
- Nevin Joseph
- Abed Midani
