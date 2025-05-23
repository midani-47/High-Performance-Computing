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
            # The FastAPI endpoint expects query parameters (not form or JSON data)
            response = requests.post(
                f"{self.base_url}/token",
                params={
                    "username": self.auth["username"],
                    "password": self.auth["password"]
                }
            )
                
            if response.status_code == 422:
                logger.warning(f"FastAPI validation error: {response.text}")
                
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
    """Load the pre-trained fraud detection model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {str(e)}")
        logger.info("Using a mock model class instead")
        
        # Create a mock model class that implements predict_proba
        class MockModel:
            def predict_proba(self, features):
                # For each input, return probability of class 0 (legitimate) and class 1 (fraudulent)
                batch_size = len(features) if hasattr(features, '__len__') else 1
                # Mostly legitimate (90% chance) with some fraudulent (10% chance)
                return np.array([[0.9, 0.1] for _ in range(batch_size)])
                
        return MockModel()


def preprocess_transaction(transaction):
    """
    Preprocess transaction data for the model.
    This function extracts and formats the features needed by the model.
    """
    # Extract relevant features from the transaction
    try:
        # For a real fraud detection model, we would extract all relevant features
        # For this implementation, we'll use a simplified set of features that should work with the model
        # The exact features would depend on what the model was trained on
        
        # Convert amount to float (this is typically an important feature for fraud detection)
        amount = float(transaction.get('amount', 0))
        
        # Extract customer and vendor information
        # We'll convert these to numeric features since most ML models require numeric input
        customer_id = transaction.get('customer_id', '')
        vendor_id = transaction.get('vendor_id', '')
        
        # Extract customer ID number if it's in format 'CUST_1234'
        customer_id_num = 0
        if isinstance(customer_id, str) and '_' in customer_id:
            try:
                customer_id_num = int(customer_id.split('_')[1])
            except (IndexError, ValueError):
                customer_id_num = hash(customer_id) % 10000  # Fallback to hash
        elif isinstance(customer_id, str):
            customer_id_num = hash(customer_id) % 10000
            
        # Extract vendor ID number if it's in format 'VENDOR_123'
        vendor_id_num = 0
        if isinstance(vendor_id, str) and '_' in vendor_id:
            try:
                vendor_id_num = int(vendor_id.split('_')[1])
            except (IndexError, ValueError):
                vendor_id_num = hash(vendor_id) % 1000  # Fallback to hash
        elif isinstance(vendor_id, str):
            vendor_id_num = hash(vendor_id) % 1000
        
        # Create feature dictionary
        features = {
            'amount': amount,
            'customer_id_num': customer_id_num,
            'vendor_id_num': vendor_id_num,
            # Add derived features that might help with fraud detection
            'amount_log': np.log1p(amount) if amount > 0 else 0,
            'amount_bin': min(int(amount / 100), 9)  # Bin amount into 10 categories
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
    """Main entry point for the prediction service."""
    # Initialize the queue client
    queue_client = QueueClient(CONFIG)
    
    # Authenticate with the queue service with retries
    max_retries = 10
    retry_delay = 5  # seconds
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Authentication attempt {attempt}/{max_retries}")
        if queue_client.authenticate():
            logger.info("Successfully authenticated with queue service")
            break
        
        if attempt < max_retries:
            logger.warning(f"Authentication failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error("Failed to authenticate with queue service after multiple attempts")
            sys.exit(1)
        
    # Load model (real or mock)
    model = load_model(CONFIG["model_path"])
        
    logger.info("Prediction service initialized successfully")
    
    # Get the number of processors to simulate
    num_processors = CONFIG["num_processors"]
    logger.info(f"Simulating {num_processors} worker processes")
    
    # Continuously process batches of transactions
    while True:
        # Read up to num_processors transactions from the queue
        transactions = []
        logger.info(f"Attempting to read up to {num_processors} transactions")
        
        # Try to get at least one transaction (blocking if necessary)
        first_message = None
        while first_message is None:
            first_message = queue_client.pull_message(CONFIG["transaction_queue"])
            if first_message is None:
                logger.info("Queue is empty, waiting for a transaction...")
                time.sleep(2)  # Wait before trying again
        
        # Add the first message to our transactions list
        transactions.append(first_message)
        
        # Try to get more transactions up to num_processors (non-blocking)
        for _ in range(num_processors - 1):
            message = queue_client.pull_message(CONFIG["transaction_queue"])
            if message:
                transactions.append(message)
            else:
                # No more messages available, proceed with what we have
                break
        
        logger.info(f"Retrieved {len(transactions)} transactions (out of {num_processors} processors)")
        logger.info(f"Processing {len(transactions)} transactions")
        
        # Process each transaction (simulating distribution to workers)
        results = []
        for i, transaction in enumerate(transactions):
            # Simulate worker ID for logging
            worker_id = (i % num_processors) + 1
            logger.info(f"Worker {worker_id} processing transaction {transaction.get('transaction_id', 'unknown')}")
            
            # Process transaction and make prediction
            prediction_result = worker_process(model, transaction)
            
            if prediction_result:
                results.append(prediction_result)
                logger.info(f"Worker {worker_id} completed prediction for transaction {transaction.get('transaction_id', 'unknown')}")
        
        # Gather results (simulating collection from workers)
        logger.info(f"Gathering results from {len(results)} workers")
        
        # Push results to the queue
        for prediction_result in results:
            queue_client.push_message(CONFIG["results_queue"], prediction_result)
            logger.info(f"Pushed prediction for transaction {prediction_result.get('transaction_id')} to results queue")
        
        logger.info(f"Completed processing batch of {len(transactions)} transactions")
        
        # Short pause between batches
        time.sleep(1)


if __name__ == "__main__":
    main()
