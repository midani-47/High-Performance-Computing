# Distributed Fraud Detection System with MPI

This project implements a distributed fraud detection system using MPI (Message Passing Interface) for parallel processing. The system reads transactions from a queue, distributes them to multiple processors for fraud prediction, and writes the results back to a prediction queue.

## Project Structure

- `fraud_detection_mpi.py`: Main MPI implementation for distributed fraud detection
- `test_fraud_detection.py`: Test script for evaluating the fraud detection model
- `fraud_detection_ui.py`: Web-based UI for submitting transactions and viewing results
- `queue_service.py`: Simple queue service implementation for transaction and prediction queues
- `DOCUMENTATION.md`: Comprehensive technical documentation
- `mpi/`: Directory containing reference code and the pre-trained model
- `a3/queue_data/`: Directory containing the queue data files (TQ1.json for transactions and PQ1.json for predictions)

## Prerequisites

- Python 3.7+
- OpenMPI
- Required Python packages: mpi4py, scikit-learn, pandas, numpy, requests, flask

## Installation

1. Install OpenMPI:
   - On macOS: `brew install open-mpi`
   - On Ubuntu/Debian: `sudo apt-get install libopenmpi-dev`
   - On Windows: Download and install from [Open MPI for Windows](https://www.open-mpi.org/software/ompi/v4.1/) or use Microsoft MPI

2. Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install mpi4py scikit-learn pandas numpy requests flask
   ```

4. Create the queue data directory:
   ```bash
   # On macOS/Linux
   mkdir -p a3/queue_data
   
   # On Windows
   mkdir a3\queue_data
   ```
   
   This directory will store the transaction queue (TQ1.json) and prediction queue (PQ1.json) files.

## Running the System

### 1. Testing the Fraud Detection Model

To test the fraud detection model without MPI or the queue service:

```bash
# On all platforms
python test_fraud_detection.py --samples 20 --verbose
```

Options:
- `--samples`: Number of test samples to process (default: 10)
- `--verbose`: Show detailed output for each transaction

### 2. Running the MPI Service

To run the MPI service with mock data (no queue service required):

```bash
# On macOS/Linux
mpiexec -n 5 python fraud_detection_mpi.py --mock
# or
mpirun -n 5 python fraud_detection_mpi.py --mock

# On Windows with Microsoft MPI
mpiexec -n 5 python fraud_detection_mpi.py --mock
```

To run the MPI service with the queue service:

```bash
# First, start the queue service (all platforms)
python queue_service.py

# In another terminal, run the MPI service
# On macOS/Linux
mpirun -n 5 python fraud_detection_mpi.py

# On Windows with Microsoft MPI
mpiexec -n 5 python fraud_detection_mpi.py
```

Options:
- `-n`: Number of MPI processes to spawn
- `--mock`: Use mock data instead of the queue service
- `--queue-url`: Specify a custom queue service URL (default: http://localhost:8000)
- `--single`: Run in single process mode (useful for debugging)
- `--test`: Run in test mode to verify functionality

### 3. Running the UI

#### Queue Service Setup:

```bash
# All platforms
python queue_service.py --host 127.0.0.1 --port 8000
```

- Verify it's running by checking http://localhost:8000/health

#### UI Configuration:

```bash
# All platforms
# For production use
python fraud_detection_ui.py --no-debug

# For development with proper debug mode
python fraud_detection_ui.py --port 5000
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
   # On macOS/Linux
   mpirun -n 5 python fraud_detection_mpi.py
   
   # On Windows
   mpiexec -n 5 python fraud_detection_mpi.py
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