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

**For manual transaction submission via UI:**
```bash
# Start with queue service integration (for manual UI transactions)
python fraud_detection_mpi.py

# With custom processor count  
python fraud_detection_mpi.py --np 8

# Single process mode (recommended for macOS)
python fraud_detection_mpi.py --single
```

**For testing only (automatic transaction generation):**
```bash
# Test mode with auto-generated transactions - NOT for UI use
python fraud_detection_mpi.py --mock

# Single process test mode
python fraud_detection_mpi.py --single --mock
```

**IMPORTANT**: 
- **Do NOT use `--mock` flag** if you want to submit transactions manually via the UI
- The `--mock` flag is only for testing and will generate transactions automatically
- For UI-based transaction submission, use the queue service integration (no `--mock` flag)

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

### IMPORTANT: Manual Transaction Submission (Recommended)

For normal operation where you submit transactions manually via the UI:

1. **Start the queue service:**
   ```bash
   python queue_service.py
   ```

2. **Start the fraud detection service (WITHOUT --mock flag):**
   ```bash
   # Option 1: Try MPI mode (will fall back to single process on macOS)
   python fraud_detection_mpi.py
   
   # Option 2: Force single process mode (recommended for macOS)
   python fraud_detection_mpi.py --single
   ```

3. **Start the UI:**
   ```bash
   python fraud_detection_ui.py
   ```

4. **Submit transactions manually** via the web UI at http://localhost:5000

### Testing Mode (Automatic Transaction Generation)

⚠️ **WARNING**: Only use this for testing, not for normal operation!

```bash
# This will auto-generate transactions - NOT for manual UI submission
python fraud_detection_mpi.py --mock

# Single process test mode
python fraud_detection_mpi.py --single --mock
```

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

1. **❌ System generates hundreds of transactions automatically**
   - **Cause**: Using `--mock` flag for UI-based operation
   - **Solution**: Remove `--mock` flag. Use `python fraud_detection_mpi.py --single` instead
   - **Rule**: Only use `--mock` for testing, never for manual UI transactions

2. **❌ All processors show rank 0**
   - **Cause**: MPI fallback to single process mode  
   - **Solution**: This is now fixed - processor ranks will show 1, 2, 3, 4 in rotation

3. **❌ MPI service stops suddenly**
   - **Cause**: System exits when no transactions are available
   - **Solution**: Fixed - system now waits patiently for manual transactions

4. **Queue service not responding**: Ensure the queue service is running on port 8000

5. **Model file not found**: The fraud detection model should be in `mpi/fraud_rf_model.pkl`

6. **Permission errors**: Ensure the `a3/queue_data/` directory exists and is writable

7. **Graceful shutdown**: Use Ctrl+C to stop the service gracefully - you'll see "^CReceived signal 2, shutting down gracefully..."

### Signal Handling

The fraud detection service supports graceful shutdown:
- **Ctrl+C (SIGINT)**: Triggers graceful shutdown with message "^CReceived signal 2, shutting down gracefully..."
- **SIGTERM**: Also triggers graceful shutdown
- All MPI processes will be properly terminated
- In-progress transactions will complete before shutdown

## Documentation

For detailed technical documentation, please refer to [DOCUMENTATION.md](DOCUMENTATION.md).