#!/usr/bin/env python
# encoding: utf8

import json
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from mpi4py import MPI
import joblib
from datetime import datetime
import argparse

# Constants
MODEL_FILENAME = 'mpi/fraud_rf_model.pkl'
STATUSES = ['submitted', 'accepted', 'rejected']
DEFAULT_PROCESSORS = 5
QUEUE_SERVICE_URL = 'http://localhost:8000'
TRANSACTION_QUEUE = 'TQ1'  # Transaction Queue name from Assignment 3
PREDICTION_QUEUE = 'PQ1'    # Prediction Queue name from Assignment 3

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_model(filename=MODEL_FILENAME):
    """Load the pre-trained fraud detection model"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    return joblib.load(filename)


def preprocess_transaction(transaction):
    """Preprocess a transaction for prediction"""
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


def predict_fraud(model, transaction):
    """Predict if a transaction is fraudulent"""
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


def fetch_transactions(num_transactions=1):
    """Fetch transactions from the queue service"""
    transactions = []
    try:
        # In a real implementation, this would use proper authentication
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


def send_prediction_result(result):
    """Send prediction result to the prediction queue"""
    try:
        # In a real implementation, this would use proper authentication
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


def mock_fetch_transactions(num_transactions=1):
    """Mock function to generate synthetic transactions for testing"""
    transactions = []
    for i in range(num_transactions):
        transaction = {
            "transaction_id": f"tx-{int(time.time())}-{i}",
            "customer_id": f"cust-{i+1}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": np.random.choice(STATUSES),
            "vendor_id": np.random.randint(1, 100),
            "amount": round(np.random.uniform(10.0, 1000.0), 2)
        }
        transactions.append(transaction)
    return transactions


def mock_send_prediction_result(result):
    """Mock function to print prediction results for testing"""
    print(f"Mock sending prediction result: {json.dumps(result, indent=2)}")
    return True


def main(use_mock=False, num_processors=DEFAULT_PROCESSORS):
    """Main function to run the fraud detection service"""
    if rank == 0:
        print(f"Starting Fraud Detection Service with {size} processors")
        print(f"Using {'mock data' if use_mock else 'real queue service'}")
        
        # Load the model (only root process needs to load it initially)
        model = load_model()
        
        while True:
            # Determine how many transactions to fetch
            batch_size = min(size, num_processors)
            
            # Fetch transactions
            if use_mock:
                transactions = mock_fetch_transactions(batch_size)
            else:
                transactions = fetch_transactions(batch_size)
            
            if not transactions:
                print("No transactions available. Waiting...")
                time.sleep(5)  # Wait before trying again
                continue
            
            print(f"Fetched {len(transactions)} transactions")
            
            # Prepare data for workers
            # Pad with None if we have fewer transactions than processors
            worker_data = transactions + [None] * (size - len(transactions))
            
            # Broadcast the model to all workers (first time only)
            model_broadcast = comm.bcast(model, root=0)
            
            # Scatter transactions to workers
            transaction = comm.scatter(worker_data, root=0)
            
            # Process own transaction (root also does work)
            result = None
            if transaction is not None:
                result = predict_fraud(model, transaction)
            
            # Gather results from all workers
            results = comm.gather(result, root=0)
            
            # Send results to prediction queue
            for result in results:
                if result is not None:
                    if use_mock:
                        mock_send_prediction_result(result)
                    else:
                        send_prediction_result(result)
            
            # Small delay to prevent CPU hogging
            time.sleep(0.1)
    else:
        # Worker processes
        while True:
            # Receive the model from root
            model = comm.bcast(None, root=0)
            
            # Receive transaction from root
            transaction = comm.scatter(None, root=0)
            
            # Process transaction if not None
            result = None
            if transaction is not None:
                result = predict_fraud(model, transaction)
            
            # Send result back to root
            comm.gather(result, root=0)
            
            # Small delay to prevent CPU hogging
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection MPI Service")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of real queue service")
    parser.add_argument("--processors", type=int, default=DEFAULT_PROCESSORS, help="Number of processors to use")
    
    # Only parse args on the root process to avoid conflicts
    args = parser.parse_args() if rank == 0 else None
    
    # Broadcast args to all processes
    if rank == 0:
        use_mock = args.mock
        num_processors = args.processors
    else:
        use_mock = None
        num_processors = None
    
    use_mock = comm.bcast(use_mock, root=0)
    num_processors = comm.bcast(num_processors, root=0)
    
    try:
        main(use_mock, num_processors)
    except KeyboardInterrupt:
        print("Shutting down Fraud Detection Service...")
        sys.exit(0)