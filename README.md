# Distributed Fraud Detection System with MPI

This project implements a distributed fraud detection system using MPI (Message Passing Interface) for parallel processing. The system reads transactions from a queue, distributes them to multiple processors for fraud prediction, and writes the results back to a prediction queue.

## Project Structure

- `fraud_detection_mpi.py`: Main MPI implementation for distributed fraud detection
- `test_fraud_detection.py`: Test script for evaluating the fraud detection model
- `fraud_detection_ui.py`: Web-based UI for submitting transactions and viewing results
- `queue_service.py`: Simple queue service implementation for transaction and prediction queues
- `DOCUMENTATION.md`: Comprehensive technical documentation
- `mpi/`: Directory containing reference code and the pre-trained model
- `a3/queue_data/`: Directory containing the queue data files

## Prerequisites

- Python 3.7+
- OpenMPI
- Required Python packages: mpi4py, scikit-learn, pandas, numpy, requests, flask

## Installation

1. Install OpenMPI:
   - On macOS: `brew install open-mpi`
   - On Ubuntu: `sudo apt-get install libopenmpi-dev`

2. Activate the virtual environment
```bash
python -m venv venv 
.\venv\Scripts\activate

#On macOS/Linux:
source venv/bin/activate
```

3. Install Python dependencies:
   ```bash
   pip install mpi4py scikit-learn pandas numpy requests flask
   ```

## Running the System

### 1. Testing the Fraud Detection Model

To test the fraud detection model without MPI or the queue service:

```bash
python test_fraud_detection.py --samples 20 --verbose
```

Options:
- `--samples`: Number of test samples to process (default: 10)
- `--verbose`: Show detailed output for each transaction

### 2. Running the MPI Service

To run the MPI service with mock data (no queue service required):

```bash
mpiexec -n 5 python fraud_detection_mpi.py --mock

# or
mpirun -n 5 python fraud_detection_mpi.py --mock
```

To run the MPI service with the queue service:

```bash
# First, start the queue service
python queue_service.py

# In another terminal, run the MPI service
mpirun -n 5 python fraud_detection_mpi.py
```

Options:
- `-n`: Number of MPI processes to spawn
- `--mock`: Use mock data instead of the queue service
- `--processors`: Number of processors to use (default: 5)

### 3. Running the UI

To run the web-based UI:

```bash
python fraud_detection_ui.py
```

The UI will be available at http://localhost:5000.

Options:
- `--port`: Port to run the UI on (default: 5000)
- `--no-browser`: Don't open browser automatically

## Testing End-to-End

To test the complete system:

1. Start the queue service:
   ```bash
   python queue_service.py
   ```

2. Start the MPI service:
   ```bash
   mpirun -n 5 python fraud_detection_mpi.py
   ```

3. Start the UI:
   ```bash
   python fraud_detection_ui.py
   ```

4. Use the UI to submit transactions and view prediction results

## Queue Service API

The queue service provides the following endpoints:

- `GET /api/queues`: List all available queues
- `POST /api/queues/<queue_name>/push`: Push a message to a queue
- `POST /api/queues/<queue_name>/pull`: Pull a message from a queue
- `GET /api/queues/<queue_name>/peek`: Peek at a queue without removing messages
- `POST /api/queues/<queue_name>/clear`: Clear all messages from a queue

All API calls require an Authorization header with a Bearer token:
```
Authorization: Bearer mock_token
```

## Documentation

For detailed technical documentation, please refer to [DOCUMENTATION.md](DOCUMENTATION.md).