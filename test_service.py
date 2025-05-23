#!/usr/bin/env python3
"""
Test script for the MPI-based ML Prediction Service

This script creates the necessary queues on the queue service from Assignment 3
and pushes sample transaction data for testing the prediction service.
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_service.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_script")

# Configuration
CONFIG = {
    "queue_service_url": os.getenv("QUEUE_SERVICE_URL", "http://localhost:7500"),
    "transaction_queue": os.getenv("TRANSACTION_QUEUE", "transactions"),
    "results_queue": os.getenv("RESULTS_QUEUE", "predictions"),
    "admin_auth": {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "password": os.getenv("ADMIN_PASSWORD", "admin_password")
    },
    "agent_auth": {
        "username": os.getenv("AUTH_USERNAME", "agent"),
        "password": os.getenv("AUTH_PASSWORD", "agent_password")
    }
}

class QueueServiceClient:
    """Client for interacting with the queue service from Assignment 3."""
    
    def __init__(self, config, use_admin=False):
        """Initialize the queue client with configuration."""
        self.base_url = config["queue_service_url"]
        self.auth = config["admin_auth"] if use_admin else config["agent_auth"]
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
            logger.info(f"Authentication successful as {self.auth['username']}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_headers(self):
        """Get authentication headers for API requests."""
        if not self.token:
            self.authenticate()
        return {"Authorization": f"{self.token_type} {self.token}"}
    
    def create_queue(self, queue_name, queue_type):
        """Create a new queue on the queue service."""
        try:
            response = requests.post(
                f"{self.base_url}/queues",
                headers=self.get_headers(),
                json={"name": queue_name, "type": queue_type}
            )
            
            if response.status_code == 409:
                logger.info(f"Queue {queue_name} already exists")
                return True
                
            response.raise_for_status()
            logger.info(f"Queue {queue_name} created successfully")
            return True
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, re-authenticate
                self.authenticate()
                return self.create_queue(queue_name, queue_type)
            logger.error(f"Error creating queue: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error creating queue: {str(e)}")
            return False
    
    def list_queues(self):
        """List all queues on the queue service."""
        try:
            response = requests.get(
                f"{self.base_url}/queues",
                headers=self.get_headers()
            )
            response.raise_for_status()
            queues = response.json().get("queues", [])
            logger.info(f"Found {len(queues)} queues")
            return queues
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, re-authenticate
                self.authenticate()
                return self.list_queues()
            logger.error(f"Error listing queues: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error listing queues: {str(e)}")
            return []
    
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
            message = response.json()
            logger.info(f"Message pulled from {queue_name} successfully")
            return message
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


def generate_sample_transaction():
    """Generate a sample transaction for testing."""
    transaction_id = str(uuid.uuid4())
    return {
        "transaction_id": transaction_id,
        "customer_id": f"CUST_{random.randint(1000, 9999)}",
        "amount": round(random.uniform(10.0, 1000.0), 2),
        "vendor_id": f"VENDOR_{random.randint(100, 999)}",
        "timestamp": time.time()
    }


def setup_queues():
    """Set up the necessary queues for testing."""
    # Use admin client to create queues
    client = QueueServiceClient(CONFIG, use_admin=True)
    if not client.authenticate():
        logger.error("Failed to authenticate with admin credentials")
        return False
    
    # Create transaction queue
    if not client.create_queue(CONFIG["transaction_queue"], "transaction"):
        logger.error(f"Failed to create transaction queue {CONFIG['transaction_queue']}")
        return False
    
    # Create prediction results queue
    if not client.create_queue(CONFIG["results_queue"], "prediction"):
        logger.error(f"Failed to create prediction queue {CONFIG['results_queue']}")
        return False
    
    logger.info("Queues set up successfully")
    return True


def push_sample_transactions(num_transactions=10):
    """Push sample transactions to the transaction queue."""
    client = QueueServiceClient(CONFIG)
    if not client.authenticate():
        logger.error("Failed to authenticate")
        return False
    
    # Generate and push transactions
    for i in range(num_transactions):
        transaction = generate_sample_transaction()
        if not client.push_message(CONFIG["transaction_queue"], transaction):
            logger.error(f"Failed to push transaction {i+1}")
            continue
        logger.info(f"Pushed transaction {i+1}: {transaction['transaction_id']}")
    
    logger.info(f"Pushed {num_transactions} sample transactions")
    return True


def check_prediction_results():
    """Check for prediction results in the results queue."""
    client = QueueServiceClient(CONFIG)
    if not client.authenticate():
        logger.error("Failed to authenticate")
        return False
    
    # Pull and display prediction results
    results = []
    while True:
        result = client.pull_message(CONFIG["results_queue"])
        if not result:
            break
        results.append(result)
        logger.info(f"Prediction result: {json.dumps(result, indent=2)}")
    
    logger.info(f"Found {len(results)} prediction results")
    return results


def main():
    """Main function to run the test."""
    # Check if queue service is running
    try:
        response = requests.get(f"{CONFIG['queue_service_url']}")
        if response.status_code == 404:
            # FastAPI returns 404 for root path if no root handler
            logger.info("Queue service is running")
        else:
            response.raise_for_status()
    except Exception as e:
        logger.error(f"Queue service is not running: {str(e)}")
        logger.error(f"Please start the queue service from Assignment 3 first")
        return
    
    # Set up queues
    if not setup_queues():
        logger.error("Failed to set up queues")
        return
    
    # Ask user how many transactions to push
    try:
        num_transactions = int(input("Enter number of sample transactions to push (default: 10): ") or 10)
    except ValueError:
        num_transactions = 10
        logger.warning("Invalid input, using default value of 10 transactions")
    
    # Push sample transactions
    if not push_sample_transactions(num_transactions):
        logger.error("Failed to push sample transactions")
        return
    
    # Inform user about prediction service
    print("\nSample transactions have been pushed to the transaction queue.")
    print("Please ensure the prediction service is running to process these transactions.")
    print("You can start the prediction service using:")
    print("  - Local mode: ./start_service.sh")
    print("  - Docker mode: docker-compose up")
    
    # Ask if user wants to check for results
    check_results = input("\nDo you want to check for prediction results? (y/n, default: y): ").lower() or "y"
    if check_results == "y":
        wait_time = int(input("How many seconds to wait before checking? (default: 5): ") or 5)
        print(f"Waiting {wait_time} seconds for predictions to be processed...")
        time.sleep(wait_time)
        results = check_prediction_results()
        
        if not results:
            print("\nNo prediction results found. Possible reasons:")
            print("  - The prediction service is not running")
            print("  - The prediction service hasn't finished processing")
            print("  - There was an error in the prediction service")
            print("\nYou can run this script again later with the check-only option to check for results.")
    
    print("\nTest script completed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        # Only check for results
        check_prediction_results()
    else:
        main()
