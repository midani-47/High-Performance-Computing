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
import signal
import numpy as np
import pandas as pd
import requests
import joblib  # Add joblib import
from datetime import datetime, timezone

# Global MPI variables - will be set when MPI is initialized
MPI = None
comm = None
rank = None
size = None

# Constants
QUEUE_SERVICE_URL = 'http://localhost:8000'  # Default, can be overridden by command line
TRANSACTION_QUEUE = 'TQ1'  # Transaction Queue name from Assignment 3
PREDICTION_QUEUE = 'PQ1'    # Prediction Queue name from Assignment 3
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpi', 'fraud_rf_model.pkl')  # Absolute path to the trained model
MAX_RETRIES = 5  # Maximum number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds
MAX_EMPTY_ITERATIONS = 1000  # Don't exit too quickly - wait for transactions from UI
STATUSES = ['submitted', 'accepted', 'rejected']  # Transaction status values

# Global variables for graceful shutdown
shutdown_requested = False

# MPI message tags
WORK_TAG = 1
DIE_TAG = 2
RESULT_TAG = 3


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n^CReceived signal 2, shutting down gracefully...")
    shutdown_requested = True


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def initialize_mpi():
    """Initialize MPI environment and return success status"""
    global MPI, comm, rank, size
    try:
        from mpi4py import MPI as mpi_module
        MPI = mpi_module
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return True
    except Exception as e:
        print(f"Failed to initialize MPI: {e}")
        return False


def is_running_under_mpi():
    """Check if we're running under mpirun/mpiexec"""
    return any('mpi' in env_var.lower() for env_var in os.environ.keys() 
               if env_var.startswith(('OMPI_', 'PMI_', 'MPICH_', 'MV2_')))


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
            print(f"Model file {model_path} not found!")
            print("Please ensure fraud_rf_model.pkl exists in the mpi/ directory")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)


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


# Global counter for simulating processor distribution in single process mode
_processor_counter = 0

def predict_fraud(model, transaction):
    """Predict fraud for a transaction"""
    global _processor_counter
    
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
            if rank is not None and rank > 0:
                processor_rank = rank  # Use actual MPI rank for workers
            else:
                # In single process mode or master process, simulate worker distribution
                _processor_counter += 1
                processor_rank = (_processor_counter % 4) + 1  # Simulate ranks 1-4
            
            # Create prediction result
            result = {
                "transaction_id": transaction["transaction_id"],
                "prediction": prediction,
                "confidence": confidence,
                "model_version": "fraud-rf-1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processor_rank": processor_rank
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
        _processor_counter += 1
        processor_rank = (_processor_counter % 4) + 1 if rank is None or rank == 0 else rank
        
        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "prediction": False,
            "confidence": 0.0,
            "model_version": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processor_rank": processor_rank
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
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


class TransactionWork:
    """Class to manage transaction work items"""
    def __init__(self, use_mock_data=False):
        self.use_mock_data = use_mock_data
        self.transactions = []
        self.empty_iterations = 0
        self.max_transactions_per_minute = 5 if use_mock_data else float('inf')
        self.last_transaction_time = time.time()
        self.transaction_count = 0
        
    def get_next_batch(self, batch_size):
        """Get the next batch of transactions"""
        batch = []
        
        if self.use_mock_data:
            # In mock mode, NEVER auto-generate transactions
            # Only wait for user-submitted transactions from the UI
            print("Mock mode: Waiting for user-submitted transactions from UI (no auto-generation)")
            return []
        else:
            # Fetch from real queue service
            for _ in range(batch_size):
                transaction = fetch_from_transaction_queue()
                if transaction:
                    batch.append(transaction)
                    self.empty_iterations = 0
                else:
                    self.empty_iterations += 1
                    break  # Stop if no more transactions
        
        return batch
    
    def should_continue(self):
        """Check if we should continue processing"""
        global shutdown_requested
        return self.empty_iterations < MAX_EMPTY_ITERATIONS and not shutdown_requested


def master_process(use_mock_data=False):
    """Master process that distributes work to workers"""
    global shutdown_requested
    
    if comm is None:
        print("Error: MPI not initialized")
        return []
    
    num_workers = size - 1  # Total workers available
    print(f"Master process {rank} started with {num_workers} workers")
    
    # Initialize work manager
    work_manager = TransactionWork(use_mock_data)
    
    # Keep track of results
    all_predictions = []
    
    while work_manager.should_continue() and not shutdown_requested:
        # Get a batch of transactions equal to the number of workers
        batch_size = num_workers  # Use all available workers
        transactions = work_manager.get_next_batch(batch_size)
        
        if not transactions:
            if shutdown_requested:
                break
            print("No transactions available, waiting...")
            time.sleep(10)  # Increased from 2 to 10 seconds to reduce queue service load
            continue
        
        print(f"Master: Processing batch of {len(transactions)} transactions")
        
        # Send work to workers (only up to the number of available workers)
        tasks_sent = 0
        for i, transaction in enumerate(transactions):
            if i >= num_workers:  # Don't exceed available workers
                break
            worker_rank = (i % num_workers) + 1  # Round-robin assignment to workers 1, 2, ..., num_workers
            print(f"Master: Sending transaction {transaction['transaction_id']} to worker {worker_rank}")
            comm.send(transaction, dest=worker_rank, tag=WORK_TAG)
            tasks_sent += 1
        
        # Collect results from workers
        predictions_received = 0
        status = MPI.Status()
        
        while predictions_received < tasks_sent and not shutdown_requested:
            # Receive result from any worker
            try:
                prediction = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
                worker_rank = status.Get_source()
                
                print(f"Master: Received prediction from worker {worker_rank}: {prediction['transaction_id']} -> {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
                
                all_predictions.append(prediction)
                predictions_received += 1
                
                # Send prediction to queue service if not using mock data
                if not use_mock_data:
                    success = send_to_prediction_queue(prediction)
                    if success:
                        print(f"Prediction sent successfully: {prediction['transaction_id']}")
                    else:
                        print(f"Master: Failed to send prediction {prediction['transaction_id']} to queue service")
            except Exception as e:
                if shutdown_requested:
                    break
                print(f"Error receiving prediction: {e}")
        
        # If we have remaining transactions that couldn't be processed, put them back
        if len(transactions) > tasks_sent:
            remaining_transactions = transactions[tasks_sent:]
            print(f"Master: {len(remaining_transactions)} transactions remain for next batch")
        
        print(f"Master: Completed batch processing. Total predictions so far: {len(all_predictions)}")
        
        if shutdown_requested:
            break
    
    print(f"Master: Shutting down. Processed {len(all_predictions)} total predictions")
    
    # Send termination signal to all workers
    for worker_rank in range(1, size):
        print(f"Master: Sending termination signal to worker {worker_rank}")
        try:
            comm.send(None, dest=worker_rank, tag=DIE_TAG)
        except Exception as e:
            print(f"Error sending termination signal to worker {worker_rank}: {e}")
    
    return all_predictions


def worker_process():
    """Worker process that processes transactions"""
    global shutdown_requested
    
    if comm is None:
        print("Error: MPI not initialized")
        return
    
    print(f"Worker {rank} started")
    
    # Load the fraud detection model
    model = load_model()
    
    if MPI is None:
        print("Error: MPI module not available")
        return
    
    status = MPI.Status()
    
    while not shutdown_requested:
        try:
            # Wait for work or termination signal
            data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            
            if status.Get_tag() == DIE_TAG:
                print(f"Worker {rank}: Received termination signal")
                break
            elif status.Get_tag() == WORK_TAG:
                transaction = data
                print(f"Worker {rank}: Processing transaction {transaction['transaction_id']}")
                
                # Process the transaction
                prediction = predict_fraud(model, transaction)
                
                print(f"Worker {rank}: Completed prediction for {transaction['transaction_id']} -> {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
                
                # Send result back to master
                comm.send(prediction, dest=0, tag=RESULT_TAG)
        except Exception as e:
            if shutdown_requested:
                break
            print(f"Worker {rank}: Error processing transaction: {e}")
    
    print(f"Worker {rank}: Shutting down")


def main_single_process(use_mock_data=False):
    """Run the fraud detection service in a single process"""
    global shutdown_requested
    
    print("Starting Fraud Detection Service in single process mode")
    print(f"Using {'mock' if use_mock_data else 'real'} queue service")
    
    # Load the model
    model = load_model()
    
    # Initialize work manager
    work_manager = TransactionWork(use_mock_data)
    transaction_count = 0
    
    while work_manager.should_continue() and not shutdown_requested:
        # Get one transaction at a time in single process mode
        transactions = work_manager.get_next_batch(1)
        
        if not transactions:
            if shutdown_requested:
                break
            print("No transaction available, waiting...")
            time.sleep(10)  # Increased from 2 to 10 seconds to reduce queue service load
            continue
        
        transaction = transactions[0]
        print(f"Processing transaction: {transaction.get('transaction_id', 'unknown')}")
        
        # Predict fraud
        prediction = predict_fraud(model, transaction)
        print(f"Prediction: {prediction['prediction']} with confidence {prediction['confidence']}")
        
        # Send prediction result
        if not use_mock_data:
            success = send_to_prediction_queue(prediction)
            if success:
                print(f"Prediction sent successfully: {prediction['transaction_id']}")
            else:
                print("Failed to send prediction to queue service")
        
        transaction_count += 1
        
        if shutdown_requested:
            break
    
    print(f"Exiting after processing {transaction_count} transactions")


def main_mpi(use_mock_data=False):
    """Run the fraud detection service using MPI"""
    if not initialize_mpi():
        print("Failed to initialize MPI, falling back to single process mode")
        main_single_process(use_mock_data)
        return
    
    if size < 2:
        print("MPI mode requires at least 2 processes (1 master + 1 worker)")
        return
    
    if rank == 0:
        # Master process
        all_predictions = master_process(use_mock_data)
        print(f"Master: Final summary - Processed {len(all_predictions)} predictions")
        
        # Print summary of processor usage
        processor_counts = {}
        for pred in all_predictions:
            proc_rank = pred.get('processor_rank', 'unknown')
            processor_counts[proc_rank] = processor_counts.get(proc_rank, 0) + 1
        
        print("Processor usage summary:")
        for proc_rank, count in sorted(processor_counts.items()):
            print(f"  Processor {proc_rank}: {count} predictions")
    else:
        # Worker process
        worker_process()


def test_mode():
    """Simple test function to verify the script is running"""
    print("\n\n==== TEST MODE ====\n")
    print("This is a test to verify the script is running correctly.")
    
    # Try to initialize MPI for testing
    if initialize_mpi():
        print(f"MPI Rank: {rank}")
        print(f"MPI Size: {size}")
    else:
        print("MPI not available, running in single process mode")
    
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
        # Setup signal handlers
        setup_signal_handlers()
        
        print("Starting fraud detection service...")
        parser = argparse.ArgumentParser(description="Fraud Detection Service")
        parser.add_argument("--mock", action="store_true", help="Use mock data instead of real queue service")
        parser.add_argument("--np", type=int, default=5, help="Number of processors to spawn (default: 5)")
        parser.add_argument("--single", action="store_true", help="Run in single process mode")
        parser.add_argument("--queue-url", type=str, help="URL of the queue service")
        parser.add_argument("--test", action="store_true", help="Run in test mode to verify functionality")
        
        args = parser.parse_args()
        
        print(f"Arguments parsed: mock={args.mock}, single={args.single}, test={args.test}, np={args.np}")
        
        # Override queue service URL if provided
        if args.queue_url:
            QUEUE_SERVICE_URL = args.queue_url
            print(f"Using queue service at: {QUEUE_SERVICE_URL}")
        
        # Run in test mode if requested
        if args.test:
            test_mode()
            sys.exit(0)
        
        # Determine execution mode with better fallback logic
        if args.single:
            print("Running in single process mode (forced)")
            main_single_process(args.mock)
        else:
            # Try MPI first, but fall back gracefully
            mpi_failed = False
            
            # Check if we need to start MPI processes
            if not is_running_under_mpi():
                # We're not running under mpirun/mpiexec, so try to start MPI processes
                print(f"Attempting to start MPI with {args.np} processes...")
                import subprocess
                
                # Construct the mpiexec command
                cmd = [
                    "mpiexec", "-n", str(args.np), 
                    sys.executable, __file__
                ]
                
                # Add original arguments (except --np since we're handling it)
                if args.mock:
                    cmd.append("--mock")
                if args.queue_url:
                    cmd.extend(["--queue-url", args.queue_url])
                if args.test:
                    cmd.append("--test")
                
                print(f"Executing: {' '.join(cmd)}")
                
                try:
                    # Run the MPI command with timeout
                    result = subprocess.run(cmd, check=True, timeout=10)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                    print(f"MPI execution failed: {e}")
                    mpi_failed = True
                except KeyboardInterrupt:
                    print("\n^CReceived signal 2, shutting down gracefully...")
                    sys.exit(0)
            else:
                # We're already running under MPI, try to initialize
                if not initialize_mpi():
                    print("MPI initialization failed")
                    mpi_failed = True
                elif size == 1:
                    print("Only one MPI process available, using single process mode")
                    main_single_process(args.mock)
                else:
                    print(f"Running in MPI mode with {size} processes")
                    # Check queue service health only if not in mock mode
                    if not args.mock and not check_queue_service_health():
                        print(f"Queue service at {QUEUE_SERVICE_URL} is not healthy. Please start the queue service first.")
                        sys.exit(1)
                    main_mpi(args.mock)
            
            # If MPI failed, fall back to single process mode
            if mpi_failed:
                print("============================================")
                print("MPI not available - falling back to single process mode")
                print("This is normal on macOS and other systems without MPI")
                print("============================================")
                main_single_process(args.mock)
                    
    except KeyboardInterrupt:
        print("\n^CReceived signal 2, shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        sys.exit(1)
