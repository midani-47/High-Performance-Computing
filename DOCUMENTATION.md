# Technical Documentation: Distributed Fraud Detection System with MPI

## Authors
- Abed Midani
- Nevin Joseph

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [MPI Implementation](#mpi-implementation)
4. [Queue Service Integration](#queue-service-integration)
5. [Fraud Detection Model](#fraud-detection-model)
6. [Testing and Evaluation](#testing-and-evaluation)
7. [User Interface](#user-interface)
8. [Deployment Instructions](#deployment-instructions)
9. [Limitations and Future Improvements](#limitations-and-future-improvements)

## Introduction

This document provides technical details about a distributed fraud detection system built using the Message Passing Interface (MPI) for parallel processing. The system is designed to efficiently process transaction data and predict potential fraudulent activities using a pre-trained machine learning model.

The documentation is structured to first provide a high-level overview of the system architecture, followed by detailed explanations of the MPI implementation, queue service integration, and the fraud detection model. Finally, we discuss testing procedures, deployment instructions, and potential future improvements.

## System Architecture

The distributed fraud detection system follows a parallel processing architecture with the following components:

1. **MPI Cluster**: A set of processors working in parallel to process transaction data and make fraud predictions.

2. **Queue Service**: A message queue system that stores transaction data and prediction results (implemented in Assignment 3).

3. **Fraud Detection Model**: A pre-trained Random Forest model that predicts whether a transaction is fraudulent.

4. **User Interface**: A web-based interface for submitting transactions and viewing prediction results.

### Technology Stack

The system is built using the following technologies:

- **Python**: The primary programming language used for implementation.
- **MPI4py**: Python bindings for the Message Passing Interface (MPI) standard.
- **OpenMPI**: An open-source implementation of the MPI standard.
- **scikit-learn**: Used for the fraud detection model (Random Forest).
- **Flask**: A lightweight web framework used for the user interface.
- **FastAPI**: Used for the queue service API (from Assignment 3).

### Data Flow

1. Transactions are submitted to the transaction queue through the UI or API.
2. The MPI root process fetches transactions from the queue.
3. Transactions are distributed to worker processes for parallel prediction.
4. Each worker process runs the prediction model on its assigned transaction.
5. Results are gathered by the root process and sent to the prediction queue.
6. The UI or API can fetch prediction results from the prediction queue.

## MPI Implementation

The MPI implementation is the core of the distributed processing system. It uses the master-worker pattern where the root process (rank 0) coordinates the work distribution and result collection.

### Process Initialization

The system initializes MPI and determines the rank and size of the communicator:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

### Work Distribution

The root process fetches transactions from the queue and distributes them to worker processes:

```python
if rank == 0:
    # Fetch transactions
    transactions = fetch_transactions(batch_size)
    
    # Pad with None if we have fewer transactions than processors
    worker_data = transactions + [None] * (size - len(transactions))
    
    # Scatter transactions to workers
    transaction = comm.scatter(worker_data, root=0)
else:
    # Worker processes receive their transaction
    transaction = comm.scatter(None, root=0)
```

### Parallel Processing

Each process (including the root) processes its assigned transaction:

```python
# Process transaction if not None
result = None
if transaction is not None:
    result = predict_fraud(model, transaction)
```

### Result Collection

The root process gathers results from all workers and sends them to the prediction queue:

```python
# Gather results from all workers
results = comm.gather(result, root=0)

if rank == 0:
    # Send results to prediction queue
    for result in results:
        if result is not None:
            send_prediction_result(result)
```

## Queue Service Integration

The system integrates with the queue service from Assignment 3, which provides endpoints for pushing and pulling messages from queues.

### Fetching Transactions

Transactions are fetched from the transaction queue using the pull endpoint:

```python
def fetch_transactions(num_transactions=1):
    transactions = []
    try:
        headers = {"Authorization": "Bearer mock_token"}
        
        for _ in range(num_transactions):
            response = requests.post(
                f"{QUEUE_SERVICE_URL}/api/queues/{TRANSACTION_QUEUE}/pull",
                headers=headers
            )
            
            if response.status_code == 200:
                message = response.json()
                if message and "content" in message:
                    transactions.append(message["content"])
            elif response.status_code == 404:
                # Queue is empty
                break
    except Exception as e:
        print(f"Error fetching transactions: {e}")
    
    return transactions
```

### Sending Prediction Results

Prediction results are sent to the prediction queue using the push endpoint:

```python
def send_prediction_result(result):
    try:
        headers = {"Authorization": "Bearer mock_token"}
        
        message = {
            "content": result,
            "message_type": "prediction"
        }
        
        response = requests.post(
            f"{QUEUE_SERVICE_URL}/api/queues/{PREDICTION_QUEUE}/push",
            headers=headers,
            json=message
        )
        
        if response.status_code == 201:
            print(f"Prediction result sent successfully: {result['transaction_id']}")
            return True
        else:
            print(f"Failed to send prediction result: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error sending prediction result: {e}")
        return False
```

## Fraud Detection Model

The system uses a pre-trained Random Forest model for fraud detection. The model is trained on synthetic transaction data with the following features:

- **timestamp**: The time of the transaction (converted to Unix timestamp)
- **status**: The status of the transaction (submitted, accepted, rejected)
- **vendor_id**: The ID of the vendor
- **amount**: The transaction amount

### Model Loading

The model is loaded from a pickle file:

```python
def load_model(filename=MODEL_FILENAME):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    return joblib.load(filename)
```

### Data Preprocessing

Before making predictions, transaction data needs to be preprocessed:

```python
def preprocess_transaction(transaction):
    # Create a DataFrame with a single row
    df = pd.DataFrame([transaction])
    
    # Convert timestamp to Unix timestamp if it's a string
    if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
    
    # Encode status using the same order as training
    if 'status' in df.columns:
        status_mapping = {status: idx for idx, status in enumerate(STATUSES)}
        df['status'] = df['status'].map(status_mapping)
    
    # Drop columns not used for prediction
    columns_to_drop = ['fraudulent', 'customer_id', 'transaction_id'] 
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df
```

### Making Predictions

The model makes predictions on preprocessed transaction data:

```python
def predict_fraud(model, transaction):
    # Preprocess the transaction
    X = preprocess_transaction(transaction)
    
    # Make prediction
    prediction = bool(model.predict(X)[0])
    confidence = float(np.max(model.predict_proba(X)[0]))
    
    # Create result object
    result = {
        "transaction_id": transaction.get("transaction_id", "unknown"),
        "prediction": prediction,
        "confidence": confidence,
        "model_version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "processor_rank": rank
    }
    
    return result
```

## Testing and Evaluation

The system includes a test script (`test_fraud_detection.py`) for evaluating the fraud detection model without requiring the MPI environment or queue service.

### Test Script

The test script loads the model and test data, then makes predictions on a sample of transactions:

```python
def main(num_samples=10, verbose=False):
    # Load the model
    model = load_model()
    
    # Load test data
    transactions = load_test_data(num_samples=num_samples)
    
    # Process each transaction
    results = []
    for transaction in transactions:
        result = predict_fraud(model, transaction)
        results.append(result)
    
    # Print summary
    fraud_count = sum(1 for r in results if r["prediction"])
    print(f"Summary: {fraud_count} fraudulent transactions detected out of {len(results)}")
```

### Command-Line Testing

The system can be tested using the following commands:

1. Test the fraud detection model:
   ```bash
   python test_fraud_detection.py --samples 20 --verbose
   ```

2. Run the MPI service with mock data:
   ```bash
   mpirun -n 5 python fraud_detection_mpi.py --mock
   ```

3. Run the MPI service with the queue service:
   ```bash
   # Start the queue service (from Assignment 3)
   cd a3/queue_service
   uvicorn app.main:app --reload
   
   # In another terminal, run the MPI service
   mpirun -n 5 python fraud_detection_mpi.py
   ```

## User Interface

The system includes a web-based user interface (`fraud_detection_ui.py`) for submitting transactions and viewing prediction results.

### Features

- Submit transactions manually or generate random transactions
- View transaction history
- Check for prediction results
- Auto-refresh for real-time updates

### Running the UI

```bash
python fraud_detection_ui.py
```

The UI will be available at http://localhost:5000.

## Deployment Instructions

To deploy the distributed fraud detection system, follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install mpi4py scikit-learn pandas numpy requests flask
   ```

2. **Install OpenMPI**:
   - On macOS: `brew install open-mpi`
   - On Ubuntu: `sudo apt-get install libopenmpi-dev`

3. **Start the Queue Service** (from Assignment 3):
   ```bash
   cd a3/queue_service
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Run the MPI Service**:
   ```bash
   mpirun -n 5 python fraud_detection_mpi.py
   ```

5. **Start the UI** (optional):
   ```bash
   python fraud_detection_ui.py
   ```

### Configuration

The system can be configured using command-line arguments:

- **MPI Service**:
  - `--mock`: Use mock data instead of the queue service
  - `--processors`: Number of processors to use (default: 5)

- **UI**:
  - `--port`: Port to run the UI on (default: 5000)
  - `--no-browser`: Don't open browser automatically

## Limitations and Future Improvements

### Current Limitations

1. **Authentication**: The current implementation uses mock authentication tokens. A real implementation would need proper authentication.

2. **Error Handling**: The system has basic error handling but could be improved for production use.

3. **Model Updates**: The model is loaded from a static file and cannot be updated dynamically.

4. **Scalability**: While MPI allows for distributed processing, the current implementation assumes all processes have access to the same file system.

### Future Improvements

1. **Dynamic Model Loading**: Implement a mechanism to update the model without restarting the service.

2. **Advanced Load Balancing**: Implement more sophisticated load balancing strategies for uneven workloads.

3. **Monitoring and Logging**: Add comprehensive monitoring and logging for better observability.

4. **Containerization**: Package the system as Docker containers for easier deployment.

5. **Performance Optimization**: Optimize the data preprocessing and prediction steps for better performance.

6. **Security Enhancements**: Implement proper authentication and encryption for production use.