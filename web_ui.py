#!/usr/bin/env python3
"""
Web UI for MPI-based ML Prediction Service

This provides a simple web interface to:
1. Create transaction and prediction queues
2. Push sample transactions
3. View prediction results
4. Monitor the MPI prediction service
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_ui.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("web_ui")

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

# Log configuration for debugging
logger.info(f"Queue Service URL: {CONFIG['queue_service_url']}")
logger.info(f"Transaction Queue: {CONFIG['transaction_queue']}")
logger.info(f"Results Queue: {CONFIG['results_queue']}")

app = Flask(__name__)

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
    
    def get_queue_info(self, queue_name):
        """Get information about a specific queue."""
        try:
            response = requests.get(
                f"{self.base_url}/queues/{queue_name}",
                headers=self.get_headers()
            )
            
            if response.status_code == 404:
                logger.warning(f"Queue {queue_name} not found")
                return None
                
            response.raise_for_status()
            queue_info = response.json()
            logger.info(f"Got info for queue {queue_name}")
            return queue_info
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, re-authenticate
                self.authenticate()
                return self.get_queue_info(queue_name)
            logger.error(f"Error getting queue info: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting queue info: {str(e)}")
            return None
    
    def push_message(self, queue_name, message):
        """Push a message to the specified queue."""
        max_retries = 3
        retry_count = 0
        retry_delay = 1.0  # Start with 1 second delay
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    f"{self.base_url}/queues/{queue_name}/push",
                    headers=self.get_headers(),
                    json={"message": message},
                    timeout=10  # Add timeout to avoid hanging requests
                )
                
                if response.status_code == 429:  # Too Many Requests
                    retry_count += 1
                    logger.warning(f"Rate limited when pushing to {queue_name}, attempt {retry_count}/{max_retries}")
                    if retry_count < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Failed to push message after {max_retries} retries due to rate limiting")
                        return False
                        
                response.raise_for_status()
                logger.info(f"Message pushed to {queue_name} successfully")
                return True
                
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    # Token expired, re-authenticate
                    self.authenticate()
                    continue  # Retry with new token
                logger.error(f"Error pushing message: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"Error pushing message: {str(e)}")
                return False
                
        return False  # Exhausted retries
    
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


@app.route('/')
def index():
    """Render the main page."""
    # Check if queue service is running
    queue_service_status = "Unknown"
    try:
        # Try to access the queue service
        # Note: In Docker, we might need to wait for the queue service to be fully up
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(f"{CONFIG['queue_service_url']}")
                if response.status_code == 404:
                    # FastAPI returns 404 for root path if no root handler
                    queue_service_status = "Running"
                    break
                else:
                    queue_service_status = "Running"
                    break
            except Exception as e:
                logger.warning(f"Attempt {retry_count+1}/{max_retries} to connect to queue service failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)  # Wait before retrying
                else:
                    queue_service_status = "Not Running"
    except Exception as e:
        logger.error(f"Error checking queue service: {str(e)}")
        queue_service_status = "Not Running"
    
    # Get queue information with retries
    admin_client = QueueServiceClient(CONFIG, use_admin=True)
    
    # Try to authenticate with multiple retries
    max_auth_retries = 10
    auth_retry_count = 0
    authenticated = False
    
    while auth_retry_count < max_auth_retries:
        try:
            if admin_client.authenticate():
                authenticated = True
                logger.info("Successfully authenticated with queue service")
                break
            else:
                logger.warning(f"Authentication attempt {auth_retry_count+1}/{max_auth_retries} failed")
        except Exception as e:
            logger.warning(f"Authentication attempt {auth_retry_count+1}/{max_auth_retries} failed: {str(e)}")
        
        auth_retry_count += 1
        if auth_retry_count < max_auth_retries:
            time.sleep(3)  # Wait before retrying
    
    if not authenticated:
        return render_template('index.html', 
                              error="Failed to authenticate with queue service after multiple attempts", 
                              queue_service_status=queue_service_status)
    
    queues = admin_client.list_queues()
    logger.info(f"Found queues: {queues}")
    
    # Check if required queues exist
    # Directly check if Queue1 and Queue2 exist by name (not case sensitive)
    transaction_queue_exists = any(q.get('name').lower() == CONFIG['transaction_queue'].lower() for q in queues)
    results_queue_exists = any(q.get('name').lower() == CONFIG['results_queue'].lower() for q in queues)
    
    # Force the values to True if we know the queues exist
    if not transaction_queue_exists and any(q.get('name') == 'Queue1' for q in queues):
        transaction_queue_exists = True
        logger.info("Forcing transaction_queue_exists to True")
        
    if not results_queue_exists and any(q.get('name') == 'Queue2' for q in queues):
        results_queue_exists = True
        logger.info("Forcing results_queue_exists to True")
    
    logger.info(f"Looking for transaction queue: {CONFIG['transaction_queue']}, exists: {transaction_queue_exists}")
    logger.info(f"Looking for results queue: {CONFIG['results_queue']}, exists: {results_queue_exists}")
    
    # Get queue sizes if they exist
    transaction_queue_size = 0
    results_queue_size = 0
    
    if transaction_queue_exists:
        queue_info = admin_client.get_queue_info(CONFIG['transaction_queue'])
        if queue_info:
            transaction_queue_size = queue_info.get('message_count', 0)
    
    if results_queue_exists:
        queue_info = admin_client.get_queue_info(CONFIG['results_queue'])
        if queue_info:
            results_queue_size = queue_info.get('message_count', 0)
    
    return render_template('index.html',
                          queue_service_status=queue_service_status,
                          transaction_queue_exists=transaction_queue_exists,
                          results_queue_exists=results_queue_exists,
                          transaction_queue_size=transaction_queue_size,
                          results_queue_size=results_queue_size,
                          transaction_queue=CONFIG['transaction_queue'],
                          results_queue=CONFIG['results_queue'])


@app.route('/setup_queues', methods=['POST'])
def setup_queues():
    """Set up the required queues."""
    admin_client = QueueServiceClient(CONFIG, use_admin=True)
    if not admin_client.authenticate():
        return jsonify({"success": False, "message": "Failed to authenticate with queue service"})
    
    # Create transaction queue
    if not admin_client.create_queue(CONFIG['transaction_queue'], "transaction"):
        return jsonify({"success": False, "message": f"Failed to create transaction queue {CONFIG['transaction_queue']}"})
    
    # Create prediction results queue
    if not admin_client.create_queue(CONFIG['results_queue'], "prediction"):
        return jsonify({"success": False, "message": f"Failed to create prediction queue {CONFIG['results_queue']}"})
    
    return jsonify({"success": True, "message": "Queues set up successfully"})


@app.route('/push_transactions', methods=['POST'])
def push_transactions():
    """Push sample transactions to the transaction queue."""
    try:
        # Get number of transactions to push
        num_transactions = int(request.form.get('num_transactions', 10))
        
        # Validate
        if num_transactions <= 0 or num_transactions > 100:
            return jsonify({"success": False, "message": "Number of transactions must be between 1 and 100"})
        
        admin_client = QueueServiceClient(CONFIG, use_admin=True)
        if not admin_client.authenticate():
            return jsonify({"success": False, "message": "Failed to authenticate with queue service"})
        
        # Push transactions with a slight delay between each to avoid rate limiting
        success_count = 0
        for i in range(num_transactions):
            transaction = generate_sample_transaction()
            logger.info(f"Pushing transaction {i+1}/{num_transactions}: {transaction}")
            
            if admin_client.push_message(CONFIG['transaction_queue'], transaction):
                success_count += 1
                
            # Add a small delay between requests to avoid rate limiting
            if i < num_transactions - 1:  # No need to delay after the last one
                time.sleep(0.5)  # 500ms delay between requests
        
        if success_count == 0:
            return jsonify({"success": False, "message": "Failed to push any transactions"})
        
        # Wait a moment to let prediction service process the transactions
        time.sleep(1)
        
        return jsonify({
            "success": True,
            "message": f"Successfully pushed {success_count} out of {num_transactions} transactions."
        })
    except Exception as e:
        logger.error(f"Error in push_transactions: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/get_results')
def get_results():
    """Get prediction results from the results queue."""
    try:
        # Pull results from the queue
        admin_client = QueueServiceClient(CONFIG, use_admin=True)
        if not admin_client.authenticate():
            return jsonify({"success": False, "message": "Failed to authenticate with queue service"})
            
        results = []
        max_results = 20  # Limit the number of results to avoid overwhelming the UI
        
        # First, check if the queue exists
        queues = admin_client.list_queues()
        queue_exists = any(q.get('name') == CONFIG['results_queue'] for q in queues)
        
        if not queue_exists:
            logger.warning(f"Results queue {CONFIG['results_queue']} doesn't exist")
            return jsonify({
                "success": True,
                "count": 0,
                "results": [],
                "message": f"Results queue {CONFIG['results_queue']} doesn't exist"
            })
        
        # Get queue info to see if there are messages
        queue_info = admin_client.get_queue_info(CONFIG['results_queue'])
        logger.info(f"Results queue info: {queue_info}")
        
        # Try to pull messages multiple times with retries
        for i in range(max_results):
            result = None
            # Try a few times with delays
            for attempt in range(3):
                result = admin_client.pull_message(CONFIG['results_queue'])
                if result:
                    break
                time.sleep(0.2)  # Short delay between attempts
                
            if result:
                logger.info(f"Retrieved result {i+1}: {result}")
                results.append(result)
            else:
                logger.info(f"No more results available after {i} pulls")
                break
                
        return jsonify({
            "success": True,
            "count": len(results),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/queue_status')
def queue_status():
    """Get the status of the queues."""
    try:
        admin_client = QueueServiceClient(CONFIG, use_admin=True)
        if not admin_client.authenticate():
            return jsonify({"success": False, "message": "Failed to authenticate with queue service"})
            
        queues = admin_client.list_queues()
        
        # Check if required queues exist
        transaction_queue_exists = any(q.get('name') == CONFIG['transaction_queue'] for q in queues)
        results_queue_exists = any(q.get('name') == CONFIG['results_queue'] for q in queues)
        
        return jsonify({
            "success": True,
            "transaction_queue": transaction_queue_exists,
            "results_queue": results_queue_exists
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/force_refresh')
def force_refresh():
    """Force refresh the queue status with caching disabled."""
    try:
        admin_client = QueueServiceClient(CONFIG, use_admin=True)
        if not admin_client.authenticate():
            return jsonify({"success": False, "message": "Failed to authenticate with queue service"})
            
        # Clear any potential caching
        queues = admin_client.list_queues()
        logger.info(f"Force refresh - Found queues: {queues}")
        
        # Check if required queues exist with explicit logging
        transaction_queue_exists = any(q.get('name') == CONFIG['transaction_queue'] for q in queues)
        results_queue_exists = any(q.get('name') == CONFIG['results_queue'] for q in queues)
        
        logger.info(f"Force refresh - Transaction queue '{CONFIG['transaction_queue']}' exists: {transaction_queue_exists}")
        logger.info(f"Force refresh - Results queue '{CONFIG['results_queue']}' exists: {results_queue_exists}")
        
        # Redirect back to main page with cache busting query parameter
        return redirect(f"/?refresh={int(time.time())}")
    except Exception as e:
        logger.error(f"Force refresh error: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Always create/update the template file to ensure it's the latest version
    template_path = os.path.join('templates', 'index.html')
    with open(template_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MPI Prediction Service UI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .json-display {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-running {
            background-color: #28a745;
        }
        .status-not-running {
            background-color: #dc3545;
        }
        .status-unknown {
            background-color: #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">MPI Prediction Service UI</h1>
        
        {% if error %}
        <div class="alert alert-danger" role="alert">
            {{ error }}
        </div>
        {% endif %}
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Service Status</h5>
                    </div>
                    <div class="card-body">
                        <p>
                            <strong>Queue Service:</strong> 
                            <span class="status-indicator 
                                {% if queue_service_status == 'Running' %}status-running
                                {% elif queue_service_status == 'Not Running' %}status-not-running
                                {% else %}status-unknown{% endif %}"></span>
                            {{ queue_service_status }}
                        </p>
                        <p>
                            <strong>Transaction Queue:</strong> 
                            <span class="status-indicator 
                                {% if transaction_queue_exists %}status-running
                                {% else %}status-not-running{% endif %}"></span>
                            {{ "Exists" if transaction_queue_exists else "Does Not Exist" }}
                            {% if transaction_queue_exists %}
                            ({{ transaction_queue_size }} messages)
                            {% endif %}
                        </p>
                        <p>
                            <strong>Results Queue:</strong> 
                            <span class="status-indicator 
                                {% if results_queue_exists %}status-running
                                {% else %}status-not-running{% endif %}"></span>
                            {{ "Exists" if results_queue_exists else "Does Not Exist" }}
                            {% if results_queue_exists %}
                            ({{ results_queue_size }} messages)
                            {% endif %}
                        </p>
                        
                        {% if not transaction_queue_exists or not results_queue_exists %}
                        <button id="setupQueuesBtn" class="btn btn-primary">Setup Queues</button>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Push Transactions</h5>
                    </div>
                    <div class="card-body">
                        <form id="pushTransactionsForm">
                            <div class="mb-3">
                                <label for="numTransactions" class="form-label">Number of Transactions</label>
                                <input type="number" class="form-control" id="numTransactions" name="num_transactions" value="10" min="1" max="100">
                            </div>
                            <button type="submit" class="btn btn-primary" {% if not transaction_queue_exists %}disabled{% endif %}>
                                Push Transactions
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5>Prediction Results</h5>
                        <button id="refreshResultsBtn" class="btn btn-sm btn-outline-primary" {% if not results_queue_exists %}disabled{% endif %}>
                            Refresh Results
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="resultsContainer">
                            <p class="text-muted">No results to display. Click "Refresh Results" to check for new predictions.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Instructions</h5>
                    </div>
                    <div class="card-body">
                        <ol>
                            <li>Ensure the queue service from Assignment 3 is running at {{ queue_service_url }}</li>
                            <li>Click "Setup Queues" to create the transaction and prediction queues if they don't exist</li>
                            <li>Start the MPI prediction service using one of these methods:
                                <ul>
                                    <li><code>./start_service.sh</code> (local mode)</li>
                                    <li><code>docker-compose up</code> (Docker mode)</li>
                                </ul>
                            </li>
                            <li>Push sample transactions to the transaction queue</li>
                            <li>Click "Refresh Results" to see prediction results as they are processed</li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Setup Queues
            const setupQueuesBtn = document.getElementById('setupQueuesBtn');
            if (setupQueuesBtn) {
                setupQueuesBtn.addEventListener('click', function() {
                    setupQueuesBtn.disabled = true;
                    setupQueuesBtn.innerHTML = 'Setting up...';
                    
                    fetch('/setup_queues', {
                        method: 'POST',
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Queues set up successfully!');
                            location.reload();
                        } else {
                            alert('Error: ' + data.message);
                            setupQueuesBtn.disabled = false;
                            setupQueuesBtn.innerHTML = 'Setup Queues';
                        }
                    })
                    .catch(error => {
                        alert('Error: ' + error);
                        setupQueuesBtn.disabled = false;
                        setupQueuesBtn.innerHTML = 'Setup Queues';
                    });
                });
            }
            
            // Push Transactions
            const pushTransactionsForm = document.getElementById('pushTransactionsForm');
            if (pushTransactionsForm) {
                pushTransactionsForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const submitBtn = pushTransactionsForm.querySelector('button[type="submit"]');
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = 'Pushing...';
                    
                    const formData = new FormData(pushTransactionsForm);
                    
                    fetch('/push_transactions', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert(data.message);
                            refreshQueueStatus();
                        } else {
                            alert('Error: ' + data.message);
                        }
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = 'Push Transactions';
                    })
                    .catch(error => {
                        alert('Error: ' + error);
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = 'Push Transactions';
                    });
                });
            }
            
            // Refresh Results
            const refreshResultsBtn = document.getElementById('refreshResultsBtn');
            if (refreshResultsBtn) {
                refreshResultsBtn.addEventListener('click', function() {
                    refreshResultsBtn.disabled = true;
                    refreshResultsBtn.innerHTML = 'Refreshing...';
                    
                    fetch('/get_results')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            const resultsContainer = document.getElementById('resultsContainer');
                            
                            if (data.count === 0) {
                                resultsContainer.innerHTML = '<p class="text-muted">No results found. The prediction service might still be processing or no transactions have been processed yet.</p>';
                            } else {
                                let html = `<p>Found ${data.count} prediction results:</p>`;
                                
                                data.results.forEach((result, index) => {
                                    html += `
                                    <div class="mb-3">
                                        <h6>Prediction ${index + 1}</h6>
                                        <div class="json-display">${JSON.stringify(result, null, 2)}</div>
                                    </div>
                                    `;
                                });
                                
                                resultsContainer.innerHTML = html;
                            }
                            
                            refreshQueueStatus();
                        } else {
                            alert('Error: ' + data.message);
                        }
                        refreshResultsBtn.disabled = false;
                        refreshResultsBtn.innerHTML = 'Refresh Results';
                    })
                    .catch(error => {
                        alert('Error: ' + error);
                        refreshResultsBtn.disabled = false;
                        refreshResultsBtn.innerHTML = 'Refresh Results';
                    });
                });
            }
            
            function refreshQueueStatus() {
                fetch('/queue_status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    }
                })
                .catch(error => {
                    console.error('Error refreshing queue status:', error);
                });
            }
        });
    </script>
</body>
</html>
            ''')
    
    # Flask is already installed in the Docker container
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=7600, debug=False)
