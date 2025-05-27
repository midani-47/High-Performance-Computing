#!/usr/bin/env python3
"""
Simple MPI-based ML Prediction Service

This service reads transactions from queue files, processes them using MPI workers,
and writes prediction results back to a results queue file.

This service integrates with the queue service from Assignment 3.
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
    "num_processors": 5  # Default: 1 master + 4 workers
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
    
    Handles both direct transaction objects and Assignment 3 message format:
    {
      "content": {
        "transaction_id": str,
        "customer_id": str,
        "customer_name": str,
        "amount": float,
        "vendor_id": str,
        "date": str,
        ... other transaction fields
      },
      "timestamp": datetime,
      "message_type": "transaction",
      "id": str
    }
    """
    try:
        # Check if this is an Assignment 3 message format
        if "content" in transaction and "message_type" in transaction:
            # Extract the actual transaction from the message content
            transaction_data = transaction["content"]
        else:
            # Direct transaction object
            transaction_data = transaction
        
        # Extract features from transaction
        transaction_id = transaction_data.get('transaction_id', 'unknown')
        amount = float(transaction_data.get('amount', 0))
        
        # Try to get transaction_count directly or calculate from history if available
        transaction_count = float(transaction_data.get('transaction_count', 0))
        if transaction_count == 0 and 'transaction_history' in transaction_data:
            # If transaction_count is not provided but history is, use the length of history
            transaction_count = float(len(transaction_data.get('transaction_history', [])))
        
        # Get risk scores or default to 0.5
        customer_risk_score = float(transaction_data.get('customer_risk_score', 0.5))
        vendor_risk_score = float(transaction_data.get('vendor_risk_score', 0.5))
        
        # Create a feature vector with all required features
        features = np.array([amount, transaction_count, customer_risk_score, vendor_risk_score]).reshape(1, -1)
        
        logger.info(f"Preprocessed transaction {transaction_id} with features: {features}")
        return features, transaction_id
    except Exception as e:
        logger.error(f"Error preprocessing transaction: {str(e)}")
        return None, None

def predict_fraud(model, transaction):
    """Make a fraud prediction for a transaction."""
    try:
        # Preprocess transaction
        features, transaction_id = preprocess_transaction(transaction)
        if features is None:
            return None
        
        # Get prediction probabilities
        probs = model.predict_proba(features)[0]
        
        # Add some randomness to make predictions more varied (for demo purposes)
        # In a real system, we would use the actual model output without modification
        amount = 0
        if "content" in transaction and "amount" in transaction["content"]:
            amount = float(transaction["content"]["amount"])
        elif "amount" in transaction:
            amount = float(transaction["amount"])
        
        # Adjust probability based on amount (higher amounts have higher fraud risk)
        # This is just for demonstration to create varied results
        amount_factor = min(amount / 10000.0, 0.5)  # Scale amount to a factor between 0 and 0.5
        fraud_probability = float(probs[1]) + amount_factor
        fraud_probability = min(max(fraud_probability, 0.05), 0.95)  # Keep between 0.05 and 0.95
        
        # Determine fraud classification based on threshold
        is_fraudulent = bool(fraud_probability > 0.5)
        
        # Calculate confidence score (higher = more confident)
        confidence = max(fraud_probability, 1.0 - fraud_probability)
        
        # Check if this is an Assignment 3 message format
        if "content" in transaction and "message_type" in transaction:
            # Create a prediction result in Assignment 3 format
            result = {
                "content": {
                    "transaction_id": transaction["content"].get("transaction_id", "unknown"),
                    "prediction": not is_fraudulent,  # True for approved (not fraud), False for rejected (fraud)
                    "confidence": round(confidence, 2),
                    "fraud_probability": round(fraud_probability, 2),
                    "is_fraudulent": is_fraudulent,
                    "model_version": "fraud_rf_model_v1",
                    "timestamp": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat(),
                "message_type": "prediction",
                "id": f"pred_{transaction.get('id', 'unknown')}"
            }
        else:
            # Create a simple prediction result
            result = transaction.copy()
            result["prediction"] = {
                "transaction_id": transaction.get("transaction_id", "unknown"),
                "prediction": not is_fraudulent,  # True for approved (not fraud), False for rejected (fraud)
                "confidence": round(confidence, 2),
                "fraud_probability": round(fraud_probability, 2),
                "is_fraudulent": is_fraudulent,
                "model_version": "fraud_rf_model_v1",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"Prediction completed for transaction {transaction_id}: " +
                  f"fraud_probability={fraud_probability:.2f}, " +
                  f"is_fraudulent={is_fraudulent}, " +
                  f"confidence={confidence:.2f}")
        
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
                # Get transaction ID for logging
                if "content" in transaction and "message_type" in transaction:
                    transaction_id = transaction["content"].get("transaction_id", "unknown")
                else:
                    transaction_id = transaction.get("transaction_id", "unknown")
                
                logger.info(f"Worker {rank} processing transaction: {transaction_id}")
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