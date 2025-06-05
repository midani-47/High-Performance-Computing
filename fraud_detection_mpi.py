#!/usr/bin/env python
# encoding: utf8

import os
import sys
import time
import json
import random
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd
import requests
import joblib  # Add joblib import
from datetime import datetime, timezone
from mpi4py import MPI

# Constants
QUEUE_SERVICE_URL = 'http://localhost:8000'  # Default, can be overridden by command line
TRANSACTION_QUEUE = 'TQ1'  # Transaction Queue name from Assignment 3
PREDICTION_QUEUE = 'PQ1'    # Prediction Queue name from Assignment 3
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpi', 'fraud_rf_model.pkl')  # Absolute path to the trained model
MAX_RETRIES = 5  # Maximum number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds
MAX_EMPTY_ITERATIONS = 10  # Maximum number of iterations with no transactions before exiting
STATUSES = ['submitted', 'accepted', 'rejected']  # Transaction status values

# MPI message tags
TAGS = {
    'EXIT': 0,  # Signal to exit
    'IDLE': 1,  # Signal to wait
    'WORK': 2,  # Signal to process work
    'RESULT': 3  # Signal with result
}


def check_queue_service_health():
    """Check if the queue service is healthy"""
    try:
        print(f"Checking queue service health at {QUEUE_SERVICE_URL}/health")
        response = requests.get(f"{QUEUE_SERVICE_URL}/health", timeout=5)
        is_healthy = response.status_code == 200
        print(f"Queue service health check result: {is_healthy}")
        return is_healthy
    except requests.exceptions.RequestException as e:
        print(f"Queue service health check failed with RequestException: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error checking queue service health: {e}")
        traceback.print_exc()
        return False


def load_model(model_path=MODEL_PATH):
    """Load the fraud detection model"""
    try:
        print(f"Attempting to load model from {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            print(f"Model loaded successfully, type: {type(model)}")
            return model
        else:
            print(f"Model file {model_path} not found, using mock model")
            return MockModel()
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return MockModel()


class MockModel:
    """A mock model for fraud detection when the real model is not available"""
    def __init__(self):
        print("Initializing mock fraud detection model")
        
    def predict(self, X):
        """Return mock predictions"""
        n_samples = len(X)
        # Generate random predictions (0 for not fraud, 1 for fraud)
        # Bias towards not fraud (80% chance)
        return np.array([0 if random.random() < 0.8 else 1 for _ in range(n_samples)])
        
    def predict_proba(self, X):
        """Return mock probability predictions"""
        n_samples = len(X)
        # Generate random probabilities for each class (not fraud, fraud)
        return np.array([[random.uniform(0.5, 0.99), random.uniform(0.01, 0.5)] for _ in range(n_samples)])


def preprocess_transaction(transaction):
    """Preprocess a transaction for the model"""
    try:
        print(f"Preprocessing transaction: {transaction}")
        # Create a DataFrame with a single row
        df = pd.DataFrame([transaction])
        print(f"Created DataFrame with columns: {df.columns.tolist()}")
        
        # Convert timestamp to Unix timestamp if it's a string
        if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
            print(f"Converting timestamp from string: {df['timestamp'].iloc[0]}")
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
            print(f"Converted timestamp to: {df['timestamp'].iloc[0]}")
        
        # Encode status using the same order as training
        if 'status' in df.columns:
            print(f"Encoding status from: {df['status'].iloc[0]}")
            status_mapping = {status: idx for idx, status in enumerate(STATUSES)}
            df['status'] = df['status'].map(status_mapping)
            print(f"Encoded status to: {df['status'].iloc[0]}")
        
        # Drop columns not used for prediction
        columns_to_drop = ['fraudulent', 'customer_id', 'transaction_id'] 
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Ensure the features are in the same order as during training
        # The model was trained on: timestamp, status, vendor_id, amount
        expected_columns = ['timestamp', 'status', 'vendor_id', 'amount']
        
        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'timestamp':
                    df[col] = int(datetime.now().timestamp())
                elif col == 'status':
                    df[col] = 0  # Default to 'submitted'
                elif col == 'vendor_id':
                    df[col] = 1  # Default to vendor 1
                elif col == 'amount':
                    df[col] = 100.0  # Default amount
        
        # Reorder columns to match training order
        df = df[expected_columns]
        
        print(f"Preprocessed features: {df.columns.tolist()}")
        print(f"Preprocessed data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error preprocessing transaction: {e}")
        traceback.print_exc()
        # Return an empty DataFrame with the expected structure
        return pd.DataFrame(columns=['timestamp', 'status', 'vendor_id', 'amount'])


def predict_fraud(model, transaction):
    """Predict fraud for a transaction"""
    try:
        # Preprocess the transaction
        print("Starting prediction process...")
        X = preprocess_transaction(transaction)
        print(f"Preprocessed data shape: {X.shape}")
        
        # Make prediction
        print(f"Model type: {type(model)}")
        print("Attempting to make prediction...")
        
        try:
            # Get prediction (True/False)
            prediction = bool(model.predict(X)[0])
            print(f"Prediction result: {prediction}")
            
            # Get confidence score
            confidence = float(np.max(model.predict_proba(X)[0]))
            print(f"Confidence: {confidence}")
            
            # Get the MPI rank for the processor
            rank = MPI.COMM_WORLD.Get_rank()
            
            # Create prediction result
            result = {
                "transaction_id": transaction["transaction_id"],
                "prediction": prediction,
                "confidence": confidence,
                "model_version": "fraud-rf-1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "processor_rank": rank
            }
            
            return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"Error predicting fraud: {e}")
        traceback.print_exc()
        # Return a default prediction
        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "prediction": False,
            "confidence": 0.0,
            "model_version": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "processor_rank": MPI.COMM_WORLD.Get_rank()
        }


def fetch_from_transaction_queue():
    """Fetch a transaction from the queue service with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            # In a real implementation, this would use proper authentication
            headers = {"Authorization": "Bearer mock_token"}
            
            response = requests.post(
                f"{QUEUE_SERVICE_URL}/api/queues/{TRANSACTION_QUEUE}/pull",
                headers=headers,
                timeout=5  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                message = response.json()
                if message and "content" in message:
                    return message["content"]
                else:
                    print("Received empty message from transaction queue")
                    return None
            elif response.status_code == 404:
                # Queue is empty
                return None
            else:
                print(f"Unexpected status code: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Giving up.")
                    return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching transaction (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Giving up.")
                return None
        except Exception as e:
            print(f"Unexpected error fetching transaction: {e}")
            traceback.print_exc()
            return None
    
    return None


def send_to_prediction_queue(prediction):
    """Send a prediction to the queue service with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            # In a real implementation, this would use proper authentication
            headers = {"Authorization": "Bearer mock_token", "Content-Type": "application/json"}
            
            message = {
                "content": prediction,
                "message_type": "prediction"
            }
            
            response = requests.post(
                f"{QUEUE_SERVICE_URL}/api/queues/{PREDICTION_QUEUE}/push",
                headers=headers,
                json=message,
                timeout=5  # Add timeout to prevent hanging
            )
            
            if response.status_code == 201:
                print(f"Prediction sent successfully: {prediction['transaction_id']}")
                return True
            else:
                print(f"Failed to send prediction: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Giving up.")
                    return False
        except requests.exceptions.RequestException as e:
            print(f"Error sending prediction (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Giving up.")
                return False
        except Exception as e:
            print(f"Unexpected error sending prediction: {e}")
            traceback.print_exc()
            return False
    
    return False


def generate_mock_transaction():
    """Generate a mock transaction for testing"""
    transaction_id = f"tx-{int(time.time())}-{random.randint(1000, 9999)}"
    customer_id = f"cust-{random.randint(1000, 9999)}"
    amount = round(random.uniform(10.0, 1000.0), 2)
    vendor_id = random.randint(1, 100)
    status = random.choice(['submitted', 'accepted', 'rejected'])
    
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "amount": amount,
        "vendor_id": vendor_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }


def main_single_process(use_mock_data=False):
    """Run the fraud detection service in a single process"""
    print("Starting Fraud Detection Service in single process mode")
    print(f"Using {'mock' if use_mock_data else 'real'} queue service")
    
    # Load the model
    model = load_model()
    
    # Process transactions
    empty_iterations = 0
    transaction_count = 0
    max_transactions_per_minute = 5  # Limit to 5 transactions per minute in mock mode
    last_transaction_time = time.time()
    
    while empty_iterations < MAX_EMPTY_ITERATIONS:
        try:
            # Get a transaction
            if use_mock_data:
                # Rate limit mock transaction generation
                current_time = time.time()
                elapsed_time = current_time - last_transaction_time
                
                # Calculate the minimum time between transactions to achieve the desired rate
                min_time_between_transactions = 60.0 / max_transactions_per_minute
                
                if elapsed_time < min_time_between_transactions:
                    # Wait until it's time to generate the next transaction
                    time.sleep(min_time_between_transactions - elapsed_time)
                
                # Generate a mock transaction
                transaction = generate_mock_transaction()
                print(f"Generated mock transaction: {transaction['transaction_id']}")
                last_transaction_time = time.time()
                transaction_count += 1
                
                # Log transaction rate
                if transaction_count % 10 == 0:
                    print(f"Generated {transaction_count} transactions so far. Rate: {max_transactions_per_minute} per minute")
            else:
                # Fetch from the queue service
                transaction = fetch_from_transaction_queue()
                
            if transaction:
                print(f"Processing transaction: {transaction.get('transaction_id', 'unknown')}")
                
                # Predict fraud
                prediction = predict_fraud(model, transaction)
                print(f"Prediction: {prediction['prediction']} with confidence {prediction['confidence']}")
                
                # Send prediction result
                if not use_mock_data:
                    success = send_to_prediction_queue(prediction)
                    if not success:
                        print("Failed to send prediction to queue service")
                
                # Reset empty iterations counter
                empty_iterations = 0
            else:
                print("No transaction available, waiting...")
                empty_iterations += 1
                time.sleep(2)  # Wait before trying again
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(2)  # Wait before trying again
    
    print(f"Exiting after {MAX_EMPTY_ITERATIONS} empty iterations")


def main_mpi(use_mock_data=False):
    """Run the fraud detection service using MPI"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        # Master process
        print(f"Starting Fraud Detection Service with {size} processes")
        print(f"Using {'mock' if use_mock_data else 'real'} queue service")
        
        # Track active workers
        active_workers = 0
        
        # Process transactions
        empty_iterations = 0
        transaction_count = 0
        max_transactions_per_minute = 5  # Limit to 5 transactions per minute in mock mode
        last_transaction_time = time.time()
        
        while empty_iterations < MAX_EMPTY_ITERATIONS:
            try:
                # Check for results from workers
                if active_workers > 0:
                    for i in range(1, size):
                        if comm.Iprobe(source=i, tag=TAGS['RESULT']):
                            prediction = comm.recv(source=i, tag=TAGS['RESULT'])
                            print(f"Received prediction from worker {i}: {prediction['prediction']} with confidence {prediction['confidence']}")
                            
                            # Send prediction result to queue service
                            if not use_mock_data:
                                success = send_to_prediction_queue(prediction)
                                if not success:
                                    print("Failed to send prediction to queue service")
                            
                            active_workers -= 1
                
                # Assign work to idle workers
                if active_workers < size - 1:
                    # Get a transaction
                    if use_mock_data:
                        # Rate limit mock transaction generation
                        current_time = time.time()
                        elapsed_time = current_time - last_transaction_time
                        
                        # Calculate the minimum time between transactions to achieve the desired rate
                        min_time_between_transactions = 60.0 / max_transactions_per_minute
                        
                        if elapsed_time < min_time_between_transactions:
                            # Wait until it's time to generate the next transaction
                            time.sleep(min_time_between_transactions - elapsed_time)
                        
                        # Generate a mock transaction
                        transaction = generate_mock_transaction()
                        print(f"Generated mock transaction: {transaction['transaction_id']}")
                        last_transaction_time = time.time()
                        transaction_count += 1
                        
                        # Log transaction rate
                        if transaction_count % 10 == 0:
                            print(f"Generated {transaction_count} transactions so far. Rate: {max_transactions_per_minute} per minute")
                    else:
                        # Fetch from the queue service
                        transaction = fetch_from_transaction_queue()
                    
                    if transaction:
                        # Find an idle worker
                        for i in range(1, size):
                            if not comm.Iprobe(source=i, tag=TAGS['RESULT']):
                                # Send work to this worker
                                comm.send(transaction, dest=i, tag=TAGS['WORK'])
                                print(f"Sent transaction {transaction.get('transaction_id', 'unknown')} to worker {i}")
                                active_workers += 1
                                break
                    else:
                        print("No transaction available, waiting...")
                        empty_iterations += 1
                        time.sleep(2)  # Wait before trying again
                else:
                    # All workers are busy, wait for results
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in master process: {e}")
                traceback.print_exc()
                time.sleep(2)  # Wait before trying again
        
        print(f"Exiting after {MAX_EMPTY_ITERATIONS} empty iterations")
        
        # Signal all workers to exit
        for i in range(1, size):
            comm.send(None, dest=i, tag=TAGS['EXIT'])
    else:
        # Worker process
        print(f"Worker {rank} started")
        
        # Load the model
        model = load_model()
        
        # Process transactions sent by the master
        while True:
            try:
                # Check for exit signal
                if comm.Iprobe(source=0, tag=TAGS['EXIT']):
                    comm.recv(source=0, tag=TAGS['EXIT'])
                    print(f"Worker {rank} received exit signal")
                    break
                
                # Check for work
                if comm.Iprobe(source=0, tag=TAGS['WORK']):
                    transaction = comm.recv(source=0, tag=TAGS['WORK'])
                    print(f"Worker {rank} received transaction: {transaction.get('transaction_id', 'unknown')}")
                    
                    # Predict fraud
                    prediction = predict_fraud(model, transaction)
                    print(f"Worker {rank} prediction: {prediction['prediction']} with confidence {prediction['confidence']}")
                    
                    # Send result back to master
                    comm.send(prediction, dest=0, tag=TAGS['RESULT'])
                else:
                    # No work available, wait
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in worker {rank}: {e}")
                traceback.print_exc()
                time.sleep(1)  # Wait before trying again
        
        print(f"Worker {rank} exiting")


def test_mode():
    """Simple test function to verify the script is running"""
    print("\n\n==== TEST MODE ====\n")
    print("This is a test to verify the script is running correctly.")
    print(f"MPI Rank: {MPI.COMM_WORLD.Get_rank()}")
    print(f"MPI Size: {MPI.COMM_WORLD.Get_size()}")
    print("\nGenerating a mock transaction:")
    transaction = generate_mock_transaction()
    print(json.dumps(transaction, indent=2))
    
    print("\nLoading model:")
    model = load_model()
    
    print("\nMaking a prediction:")
    prediction = predict_fraud(model, transaction)
    print(json.dumps(prediction, indent=2))
    
    print("\n==== TEST COMPLETE ====\n")


if __name__ == "__main__":
    import traceback
    
    try:
        print("Starting fraud detection service...")
        parser = argparse.ArgumentParser(description="Fraud Detection Service")
        parser.add_argument("--mock", action="store_true", help="Use mock data instead of real queue service")
        parser.add_argument("--np", type=int, default=4, help="Number of processors to use (for MPI mode)")
        parser.add_argument("--single", action="store_true", help="Run in single process mode")
        parser.add_argument("--queue-url", type=str, help="URL of the queue service")
        parser.add_argument("--test", action="store_true", help="Run in test mode to verify functionality")
        
        args = parser.parse_args()
        
        print(f"Arguments parsed: mock={args.mock}, single={args.single}, test={args.test}")
        
        # Override queue service URL if provided
        if args.queue_url:
            # Modify the module-level variable directly
            # No need for global declaration in the main block
            QUEUE_SERVICE_URL = args.queue_url
            print(f"Using queue service at: {QUEUE_SERVICE_URL}")
        
        # Run in test mode if requested
        if args.test:
            test_mode()
            sys.exit(0)
        
        # Check if queue service is healthy
        if not args.mock and not check_queue_service_health():
            print(f"Queue service at {QUEUE_SERVICE_URL} is not healthy. Please start the queue service first.")
            sys.exit(1)
        
        # Run in single process mode if requested or if only one process is available
        if args.single or MPI.COMM_WORLD.Get_size() == 1:
            print("Running in single process mode")
            main_single_process(args.mock)
        else:
            print("Running in MPI mode")
            main_mpi(args.mock)
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()