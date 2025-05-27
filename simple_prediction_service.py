#!/usr/bin/env python3
"""
Simple MPI-based ML Prediction Service

This service reads transactions from queue files, processes them using MPI workers,
and writes prediction results back to a results queue file.
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
from datetime import datetime
from mpi4py import MPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prediction_service")

# Configuration
CONFIG = {
    "queue_data_dir": "./a3/queue_service/queue_data",
    "transaction_queue_file": "TQ1.json",
    "transaction_queue_file2": "TQ2.json",
    "results_queue_file": "PQ1.json",
    "model_path": "./mpi/fraud_rf_model.pkl",
    "num_processors": 3  # Default: 1 master + 2 workers
}

# Override config from environment variables if present
for key in CONFIG:
    env_var = key.upper()
    if env_var in os.environ:
        CONFIG[key] = os.environ[env_var]
        if key == "num_processors":
            CONFIG[key] = int(os.environ[env_var])

class SimpleQueueClient:
    """Simple client for interacting with queue files."""
    
    def __init__(self, config):
        """Initialize the queue client with configuration."""
        self.queue_data_dir = config["queue_data_dir"]
        self.transaction_queue_file = os.path.join(self.queue_data_dir, config["transaction_queue_file"])
        self.transaction_queue_file2 = os.path.join(self.queue_data_dir, config["transaction_queue_file2"])
        self.results_queue_file = os.path.join(self.queue_data_dir, config["results_queue_file"])
        
        # Ensure the queue data directory exists
        os.makedirs(self.queue_data_dir, exist_ok=True)
        
        # Initialize queue files if they don't exist
        self._initialize_queue_file(self.transaction_queue_file)
        self._initialize_queue_file(self.transaction_queue_file2)
        self._initialize_queue_file(self.results_queue_file)
    
    def _initialize_queue_file(self, file_path):
        """Initialize a queue file if it doesn't exist."""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
    
    def _read_queue(self, queue_file):
        """Read messages from a queue file."""
        try:
            with open(queue_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding JSON from {queue_file}, initializing empty queue")
            with open(queue_file, 'w') as f:
                json.dump([], f)
            return []
        except Exception as e:
            logger.error(f"Error reading queue file {queue_file}: {str(e)}")
            return []
    
    def _write_queue(self, queue_file, messages):
        """Write messages to a queue file."""
        try:
            with open(queue_file, 'w') as f:
                json.dump(messages, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing to queue file {queue_file}: {str(e)}")
            return False
    
    def get_queue_size(self, queue_file):
        """Get the number of messages in a queue."""
        try:
            messages = self._read_queue(queue_file)
            return len(messages)
        except Exception as e:
            logger.error(f"Error getting queue size for {queue_file}: {str(e)}")
            return 0
    
    def pull_messages(self, queue_file, count=1):
        """Pull multiple messages from the specified queue file."""
        try:
            # Read current messages
            messages = self._read_queue(queue_file)
            
            if not messages:
                return []
            
            # Take the requested number of messages (or all if fewer available)
            count = min(count, len(messages))
            pulled_messages = messages[:count]
            remaining_messages = messages[count:]
            
            # Write remaining messages back to file
            self._write_queue(queue_file, remaining_messages)
            
            return pulled_messages
        except Exception as e:
            logger.error(f"Error pulling messages from {queue_file}: {str(e)}")
            return []
    
    def push_message(self, queue_file, message):
        """Push a message to the specified queue file."""
        try:
            # Read current messages
            messages = self._read_queue(queue_file)
            
            # Add new message
            messages.append(message)
            
            # Write back to file
            success = self._write_queue(queue_file, messages)
            if success:
                logger.info(f"Message pushed to {os.path.basename(queue_file)} successfully")
            return success
        except Exception as e:
            logger.error(f"Error pushing message to {queue_file}: {str(e)}")
            return False

def load_model(model_path):
    """Load the pre-trained fraud detection model."""
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {str(e)}")
        logger.info("Creating a mock model for testing")
        
        # Create a mock model class for testing
        class MockModel:
            def predict_proba(self, features):
                # Always return a low probability of fraud (0.1) for testing
                return np.array([[0.9, 0.1]] * len(features))
            
            def predict(self, features):
                # Always predict not fraud (0) for testing
                return np.zeros(len(features))
        
        return MockModel()

def preprocess_transaction(transaction):
    """
    Preprocess transaction data for the model.
    Extract and format the features needed by the model.
    """
    try:
        # Extract features from transaction
        amount = float(transaction.get('amount', 0))
        transaction_count = float(transaction.get('transaction_count', 0))
        customer_risk_score = float(transaction.get('customer_risk_score', 0.5))
        vendor_risk_score = float(transaction.get('vendor_risk_score', 0.5))
        
        # Create a feature vector with all required features
        features = np.array([amount, transaction_count, customer_risk_score, vendor_risk_score]).reshape(1, -1)
        
        return features
    except Exception as e:
        logger.error(f"Error preprocessing transaction: {str(e)}")
        return None

def predict_fraud(model, transaction):
    """Make a fraud prediction for a transaction."""
    try:
        # Preprocess transaction
        features = preprocess_transaction(transaction)
        if features is None:
            return None
        
        # Get prediction probabilities
        probs = model.predict_proba(features)[0]
        
        # Add prediction to transaction
        result = transaction.copy()
        result["prediction"] = {
            "fraud_probability": float(probs[1]),  # Probability of class 1 (fraud)
            "is_fraudulent": bool(probs[1] > 0.5),  # Classification based on threshold
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed for transaction {result.get('transaction_id')}: " +
                  f"fraud_probability={result['prediction']['fraud_probability']:.4f}, " +
                  f"is_fraudulent={result['prediction']['is_fraudulent']}")
        
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

def main():
    """Main entry point for the prediction service."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Processor rank (0 is master, others are workers)
    size = comm.Get_size()  # Total number of processors
    
    num_workers = size - 1 if size > 1 else 1  # Number of worker processes
    
    logger.info(f"MPI Rank: {rank}, Size: {size}, Workers: {num_workers}")
    
    # Initialize queue client
    queue_client = SimpleQueueClient(CONFIG)
    
    # Load the model
    model_path = CONFIG["model_path"]
    
    if rank == 0:  # Master process
        logger.info("Starting master process")
        
        # Main processing loop
        while True:
            # Check both transaction queues
            tq1_size = queue_client.get_queue_size(queue_client.transaction_queue_file)
            tq2_size = queue_client.get_queue_size(queue_client.transaction_queue_file2)
            
            logger.info(f"Current queue sizes - TQ1: {tq1_size}, TQ2: {tq2_size}")
            
            # Choose the queue with more messages, or TQ1 if they're equal
            if tq1_size >= tq2_size and tq1_size > 0:
                queue_file = queue_client.transaction_queue_file
                queue_size = tq1_size
            elif tq2_size > 0:
                queue_file = queue_client.transaction_queue_file2
                queue_size = tq2_size
            else:
                logger.info("Both queues are empty, waiting...")
                time.sleep(2)  # Wait before checking again
                continue
            
            # Determine how many transactions to pull (up to the number of workers)
            batch_size = min(queue_size, num_workers)
            logger.info(f"Pulling {batch_size} transactions from {os.path.basename(queue_file)}")
            
            # Pull transactions from the queue
            transactions = queue_client.pull_messages(queue_file, batch_size)
            
            if not transactions:
                logger.info("Queue is empty, waiting for transactions...")
                time.sleep(2)  # Wait before checking again
                continue
            
            logger.info(f"Retrieved {len(transactions)} transactions for processing")
            
            # Distribute transactions to workers
            results = []
            
            # Send transactions to workers
            for i, transaction in enumerate(transactions):
                worker_rank = (i % num_workers) + 1  # Workers start at rank 1
                if worker_rank < size:  # Make sure we don't send to non-existent workers
                    logger.info(f"Sending transaction to worker {worker_rank}")
                    comm.send(transaction, dest=worker_rank)
            
            # Send None to any workers that didn't get a transaction
            for i in range(len(transactions), num_workers):
                worker_rank = i + 1
                if worker_rank < size:  # Make sure we don't send to non-existent workers
                    comm.send(None, dest=worker_rank)
            
            # Collect results from workers
            for i in range(min(len(transactions), num_workers)):
                worker_rank = (i % num_workers) + 1
                if worker_rank < size:  # Make sure we don't receive from non-existent workers
                    result = comm.recv(source=worker_rank)
                    if result is not None:
                        logger.info(f"Received valid result from worker {worker_rank}")
                        results.append(result)
                    else:
                        logger.warning(f"Worker {worker_rank} returned None result")
            
            # Push results to results queue
            if results:
                logger.info(f"Pushing {len(results)} results to the results queue")
                for result in results:
                    success = queue_client.push_message(queue_client.results_queue_file, result)
                    if not success:
                        logger.error(f"Failed to push result to queue")
                logger.info(f"Successfully pushed {len(results)} results to queue")
            else:
                logger.warning("No valid results received from workers")
            
            # Small delay to prevent CPU overuse
            time.sleep(0.5)
    
    else:  # Worker processes
        model = load_model(CONFIG["model_path"])
        if model is None:
            logger.error(f"Worker {rank}: Failed to load model, exiting")
            sys.exit(1)
        
        logger.info(f"Worker {rank} loaded model successfully")
        
        # Worker processing loop
        while True:
            # Receive transaction from master
            transaction = comm.recv(source=0)
            
            # None means no work to do in this round
            if transaction is None:
                logger.info(f"Worker {rank} received no work this round")
                comm.send(None, dest=0)
                continue
            
            # Process the transaction
            try:
                logger.info(f"Worker {rank} processing transaction: {transaction.get('transaction_id')}")
                result = predict_fraud(model, transaction)
                if result is None:
                    logger.error(f"Worker {rank} failed to process transaction")
                    comm.send(None, dest=0)
                    continue
                
                logger.info(f"Worker {rank} completed prediction")
                # Send result back to master
                comm.send(result, dest=0)
            except Exception as e:
                logger.error(f"Worker {rank} error processing transaction: {str(e)}")
                comm.send(None, dest=0)  # Send None to indicate error

if __name__ == "__main__":
    main() 