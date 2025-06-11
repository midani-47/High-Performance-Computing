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
   pip install -r requirements.txt
   ```



## Running the System

### 1. Running queue service

To run the MPI service with the queue service:

```bash
# First, start the queue service (all platforms)
python queue_service.py
```

### 2. Running fraud detection service (Default: 5 processors)

```bash
# Default: Start with 5 processors (1 master + 4 workers)
python fraud_detection_mpi.py

# Custom number of processors
python fraud_detection_mpi.py --np 8

# With mock data (for testing)
python fraud_detection_mpi.py --mock

# Single process mode (no MPI)
python fraud_detection_mpi.py --single
```

**Important**: The system defaults to 5 processors when you run `python fraud_detection_mpi.py`. You don't need to use `mpiexec -n 5` anymore.

Options:
- `--np`: Number of MPI processes to spawn (default: 5)
- `--mock`: Use mock data instead of queue service
- `--single`: Force single process mode
- `--queue-url`: Custom queue service URL

### 3. Running the UI

```bash
# For development with proper debug mode
python fraud_detection_ui.py
```

The UI will be available at http://localhost:5000.

Options:
- `--port`: Port to run the UI on (default: 5000)
- `--no-browser`: Don't open browser automatically

## Testing End-to-End

To test the complete system:

1. **For systems with properly configured MPI:**
   ```bash
   # Start the queue service
   python queue_service.py
   
   # In another terminal, start the fraud detection service (default 5 processors)
   python fraud_detection_mpi.py
   
   # Or with custom processor count
   python fraud_detection_mpi.py --np 8
   
   # In a third terminal, start the UI
   python fraud_detection_ui.py
   ```

2. **For testing without queue service:**
   ```bash
   # Test with mock data
   python fraud_detection_mpi.py --mock
   
   # Test single process mode
   python fraud_detection_mpi.py --single --mock
   ```

3. Use the UI to submit transactions and view prediction results

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

## Troubleshooting

### MPI Issues on macOS

If you encounter MPI initialization errors like:
```
It looks like MPI_INIT failed for some reason...
PML add procs failed
```

**This is expected behavior**: The system will automatically fall back to single process mode and continue working correctly. The fraud detection functionality is preserved.

**Preferred approach on macOS**:
```bash
# Single process mode (recommended for macOS)
python fraud_detection_mpi.py --single --mock
```

**For systems with working MPI** (Linux/Windows):
```bash
# Will automatically start 5 processes
python fraud_detection_mpi.py --mock

# Custom processor count
python fraud_detection_mpi.py --np 8 --mock
```

### Common Issues

1. **Queue service not responding**: Ensure the queue service is running on port 8000
2. **Model file not found**: The fraud detection model should be in `mpi/fraud_rf_model.pkl`
3. **Permission errors**: Ensure the `a3/queue_data/` directory exists and is writable
4. **Graceful shutdown**: Use Ctrl+C to stop the service gracefully - you'll see "^CReceived signal 2, shutting down gracefully..."

### Signal Handling

The fraud detection service supports graceful shutdown:
- **Ctrl+C (SIGINT)**: Triggers graceful shutdown with message "^CReceived signal 2, shutting down gracefully..."
- **SIGTERM**: Also triggers graceful shutdown
- All MPI processes will be properly terminated
- In-progress transactions will complete before shutdown

## Documentation

For detailed technical documentation, please refer to [DOCUMENTATION.md](DOCUMENTATION.md).