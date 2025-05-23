#!/usr/bin/env python3
"""
MPI-based ML Prediction Service for Fraud Detection

This service reads transactions from a queue, distributes them to MPI worker
processes for prediction using a pre-trained model, and sends the results
back to a results queue.
"""

import os
import sys
import json
import time
import pickle
import logging
import requests
from mpi4py import MPI
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_service.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prediction_service")

# Configuration
CONFIG = {
    "num_processors": int(os.getenv("NUM_PROCESSORS", 5)),
    "queue_service_url": os.getenv("QUEUE_SERVICE_URL", "http://localhost:7500"),
    "transaction_queue": os.getenv("TRANSACTION_QUEUE", "transactions"),
    "results_queue": os.getenv("RESULTS_QUEUE", "predictions"),
    "model_path": os.getenv("MODEL_PATH", "./mpi/fraud_rf_model.pkl"),
    "auth": {
        "username": os.getenv("AUTH_USERNAME", "agent"),
        "password": os.getenv("AUTH_PASSWORD", "agent_password")
    }
}

class QueueClient:
    """Client for interacting with the queue service from Assignment 3."""
    
    def __init__(self, config):
        """Initialize the queue client with configuration."""
        self.base_url = config["queue_service_url"]
        self.auth = config["auth"]
        self.token = None
        self.token_type = None
        
    def authenticate(self):
        """Authenticate with the queue service."""
        try:
            response = requests.post(
                f"{self.base_url}/token",
                data={"username": self.auth["username"], "password": self.auth["password"]}
            )
            response.raise_for_status()
            auth_data = response.json()
            self.token = auth_data["access_token"]
            self.token_type = auth_data["token_type"]
            logger.info("Authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_headers(self):
        """Get authentication headers for API requests."""
        if not self.token:
            self.authenticate()
        return {"Authorization": f"{self.token_type} {self.token}"}
    
    def pull_message(self, queue_name):
        """Pull a message from the specified queue."""
        try:
            response = requests.get(
                f"{self.base_url}/queues/{queue_name}/pull",
                headers=self.get_headers()
            )
            
            if response.status_code == 404:
                # Queue might not exist
                logger.warning(f"Queue {queue_name} not found")
                return None
                
            if response.status_code == 204:
                # Queue is empty
                logger.info(f"Queue {queue_name} is empty")
                return None
                
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, re-authenticate
                self.authenticate()
                return self.pull_message(queue_name)
            logger.error(f"Error pulling message: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error pulling message: {str(e)}")
            return None
    
    def push_message(self, queue_name, message):
        """Push a message to the specified queue."""
        try:
            response = requests.post(
                f"{self.base_url}/queues/{queue_name}/push",
                headers=self.get_headers(),
                json=message
            )
            
            if response.status_code == 404:
                # Queue might not exist
                logger.warning(f"Queue {queue_name} not found")
                return False
                
            response.raise_for_status()
            logger.info(f"Message pushed to {queue_name} successfully")
            return True
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, re-authenticate
                self.authenticate()
                return self.push_message(queue_name, message)
            logger.error(f"Error pushing message: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error pushing message: {str(e)}")
            return False


def load_model(model_path):
    """Load the pre-trained ML model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def preprocess_transaction(transaction):
    """
    Preprocess transaction data for the model.
    This function should extract and format the features needed by the model.
    """
    # Extract relevant features from the transaction
    # This will depend on your specific model requirements
    # For demonstration, we'll assume the model expects these features
    try:
        features = {
            'amount': float(transaction.get('amount', 0)),
            'customer_id': str(transaction.get('customer_id', '')),
            'vendor_id': str(transaction.get('vendor_id', ''))
        }
        
        # Convert to DataFrame for model input
        df = pd.DataFrame([features])
        return df
    except Exception as e:
        logger.error(f"Error preprocessing transaction: {str(e)}")
        return None


def make_prediction(model, transaction_data):
    """
    Make a fraud prediction using the model.
    Returns a prediction result dict with transaction_id, prediction, and confidence.
    """
    try:
        # Make prediction
        prediction_prob = model.predict_proba(transaction_data)[0]
        prediction = model.predict(transaction_data)[0]
        
        # Get confidence score (probability of the predicted class)
        confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
        
        return {
            "prediction": bool(prediction),
            "confidence": float(confidence),
            "model_version": "1.0.0",  # Include model version
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None


def worker_process(model, transaction):
    """Worker process for making predictions."""
    try:
        # Preprocess transaction data
        transaction_data = preprocess_transaction(transaction)
        if transaction_data is None:
            return None
            
        # Make prediction
        prediction_result = make_prediction(model, transaction_data)
        if prediction_result is None:
            return None
            
        # Add transaction_id to the prediction result
        prediction_result["transaction_id"] = transaction.get("transaction_id", "unknown")
        
        return prediction_result
    except Exception as e:
        logger.error(f"Error in worker process: {str(e)}")
        return None


def main():
    """Main function for the prediction service."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check if we have at least one worker
    if size < 2:
        logger.error("This application requires at least 2 MPI processes (1 master + 1 worker)")
        return
        
    # Adjust worker count based on available processes
    num_workers = min(CONFIG["num_processors"], size - 1)
    logger.info(f"Starting with {num_workers} worker processes")
    
    # Root process (master) coordinates the work
    if rank == 0:
        # Initialize queue client
        queue_client = QueueClient(CONFIG)
        if not queue_client.authenticate():
            logger.error("Failed to authenticate with queue service")
            comm.Abort(1)
            return
            
        # Load model
        model = load_model(CONFIG["model_path"])
        if model is None:
            logger.error("Failed to load model")
            comm.Abort(1)
            return
            
        logger.info("Master process initialized successfully")
        
        # Continuously process batches of transactions
        while True:
            # Read up to num_workers transactions from the queue
            transactions = []
            logger.info(f"Attempting to read up to {num_workers} transactions")
            
            for _ in range(num_workers):
                # Pull message from transaction queue
                message = queue_client.pull_message(CONFIG["transaction_queue"])
                if message:
                    transactions.append(message)
                
            # If no transactions were found, wait and try again
            if not transactions:
                logger.info("No transactions available, waiting 5 seconds...")
                time.sleep(5)
                continue
                
            logger.info(f"Processing {len(transactions)} transactions")
            
            # Distribute transactions to workers
            for i, transaction in enumerate(transactions):
                # Send to worker (use worker_id = i + 1 since rank 0 is the master)
                worker_id = (i % num_workers) + 1
                if worker_id < size:  # Ensure we don't exceed available processes
                    comm.send(transaction, dest=worker_id)
                    logger.info(f"Sent transaction {transaction.get('transaction_id', 'unknown')} to worker {worker_id}")
            
            # Send None to signal end of batch
            for worker_id in range(1, num_workers + 1):
                if worker_id < size:
                    comm.send(None, dest=worker_id)
            
            # Collect results from workers
            for i in range(len(transactions)):
                worker_id = (i % num_workers) + 1
                if worker_id < size:
                    prediction_result = comm.recv(source=worker_id)
                    logger.info(f"Received result from worker {worker_id}")
                    
                    if prediction_result:
                        # Push prediction to results queue
                        queue_client.push_message(CONFIG["results_queue"], prediction_result)
                        logger.info(f"Pushed prediction for transaction {prediction_result.get('transaction_id')} to results queue")
            
            logger.info(f"Completed processing batch of {len(transactions)} transactions")
    
    # Worker processes
    else:
        # Load model
        model = load_model(CONFIG["model_path"])
        if model is None:
            logger.error(f"Worker {rank}: Failed to load model")
            return
            
        logger.info(f"Worker {rank} initialized successfully")
        
        # Process transactions sent by the master
        while True:
            # Receive transaction from master
            transaction = comm.recv(source=0)
            
            # Check if this is the end of the batch
            if transaction is None:
                break
                
            logger.info(f"Worker {rank} processing transaction {transaction.get('transaction_id', 'unknown')}")
            
            # Process transaction and make prediction
            prediction_result = worker_process(model, transaction)
            
            # Send result back to master
            comm.send(prediction_result, dest=0)
            logger.info(f"Worker {rank} sent result back to master")


if __name__ == "__main__":
    main()
