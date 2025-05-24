#!/usr/bin/env python3
"""
Web UI for MPI-based ML Prediction Service

This provides a simple web interface to:
1. View transaction queues
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
    "queue_data_dir": os.getenv("QUEUE_DATA_DIR", "./a3/queue_service/queue_data"),
    "transaction_queue_file": os.getenv("TRANSACTION_QUEUE_FILE", "TQ1.json"),
    "transaction_queue_file2": os.getenv("TRANSACTION_QUEUE_FILE2", "TQ2.json"),
    "results_queue_file": os.getenv("RESULTS_QUEUE_FILE", "PQ1.json"),
    "model_path": os.getenv("MODEL_PATH", "./mpi/fraud_rf_model.pkl"),
    "num_processors": int(os.getenv("NUM_PROCESSORS", "5"))
}

# Log configuration for debugging
logger.info(f"Queue Data Directory: {CONFIG['queue_data_dir']}")
logger.info(f"Transaction Queue Files: {CONFIG['transaction_queue_file']}, {CONFIG['transaction_queue_file2']}")
logger.info(f"Results Queue File: {CONFIG['results_queue_file']}")

app = Flask(__name__)

class FileQueueClient:
    """Client for interacting with queue files directly."""
    
    def __init__(self, config):
        """Initialize the file queue client with configuration."""
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
        
        logger.info(f"FileQueueClient initialized with data directory: {self.queue_data_dir}")
        logger.info(f"Transaction queue files: {self.transaction_queue_file}, {self.transaction_queue_file2}")
        logger.info(f"Results queue file: {self.results_queue_file}")
    
    def _initialize_queue_file(self, file_path):
        """Initialize a queue file if it doesn't exist."""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
            logger.info(f"Created empty queue file: {file_path}")
    
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
    
    def queue_exists(self, queue_file):
        """Check if a queue file exists."""
        return os.path.exists(queue_file) and os.path.getsize(queue_file) > 0
    
    def push_message(self, queue_file, message):
        """Push a message to the specified queue file."""
        try:
            # Read current messages
            messages = self._read_queue(queue_file)
            
            # Add new message with metadata
            message_with_metadata = {
                "id": str(uuid.uuid4()),
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            
            messages.append(message_with_metadata)
            
            # Write back to file
            success = self._write_queue(queue_file, messages)
            if success:
                logger.info(f"Message pushed to {os.path.basename(queue_file)} successfully")
            return success
        except Exception as e:
            logger.error(f"Error pushing message to {queue_file}: {str(e)}")
            return False
    
    def pull_messages(self, queue_file, count=1, remove=True):
        """Pull multiple messages from the specified queue file."""
        try:
            # Read current messages
            messages = self._read_queue(queue_file)
            
            if not messages:
                logger.info(f"Queue {os.path.basename(queue_file)} is empty")
                return []
            
            # Take the requested number of messages (or all if fewer available)
            count = min(count, len(messages))
            pulled_messages = messages[:count]
            
            if remove:
                # Write remaining messages back to file
                remaining_messages = messages[count:]
                self._write_queue(queue_file, remaining_messages)
            
            # Return the full message objects
            return pulled_messages
        except Exception as e:
            logger.error(f"Error pulling messages from {queue_file}: {str(e)}")
            return []
    
    def get_queue_size(self, queue_file):
        """Get the number of messages in a queue."""
        try:
            messages = self._read_queue(queue_file)
            return len(messages)
        except Exception as e:
            logger.error(f"Error getting queue size for {queue_file}: {str(e)}")
            return 0

def generate_sample_transaction():
    """Generate a sample transaction for testing."""
    transaction_id = f"tx{random.randint(1000, 9999)}"
    customer_id = f"cust{random.randint(100, 999)}"
    customer_name = random.choice(["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"])
    amount = round(random.uniform(10, 10000), 2)
    vendor_id = f"vend{random.randint(100, 999)}"
    date = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "amount": amount,
        "vendor_id": vendor_id,
        "date": date
    }

# Initialize the file queue client
queue_client = FileQueueClient(CONFIG)

@app.route('/')
def index():
    """Render the main page."""
    # Check if queue files exist
    transaction_queue_exists = queue_client.queue_exists(queue_client.transaction_queue_file)
    transaction_queue2_exists = queue_client.queue_exists(queue_client.transaction_queue_file2)
    results_queue_exists = queue_client.queue_exists(queue_client.results_queue_file)
    
    # Get queue sizes
    transaction_queue_size = queue_client.get_queue_size(queue_client.transaction_queue_file)
    transaction_queue2_size = queue_client.get_queue_size(queue_client.transaction_queue_file2)
    results_queue_size = queue_client.get_queue_size(queue_client.results_queue_file)
    
    return render_template('index.html',
                          transaction_queue_exists=transaction_queue_exists,
                          transaction_queue2_exists=transaction_queue2_exists,
                          results_queue_exists=results_queue_exists,
                          transaction_queue_size=transaction_queue_size,
                          transaction_queue2_size=transaction_queue2_size,
                          results_queue_size=results_queue_size,
                          transaction_queue=CONFIG['transaction_queue_file'],
                          transaction_queue2=CONFIG['transaction_queue_file2'],
                          results_queue=CONFIG['results_queue_file'])

@app.route('/push_transactions', methods=['POST'])
def push_transactions():
    """Push sample transactions to the transaction queue."""
    try:
        # Get number of transactions to push
        num_transactions = int(request.form.get('num_transactions', 10))
        
        # Validate
        if num_transactions <= 0 or num_transactions > 100:
            return jsonify({"success": False, "message": "Number of transactions must be between 1 and 100"})
        
        # Get target queue (TQ1 or TQ2)
        target_queue = request.form.get('target_queue', 'TQ1')
        if target_queue == 'TQ1':
            queue_file = queue_client.transaction_queue_file
        else:
            queue_file = queue_client.transaction_queue_file2
        
        # Push transactions with a slight delay between each to avoid file locking issues
        success_count = 0
        for i in range(num_transactions):
            transaction = generate_sample_transaction()
            logger.info(f"Pushing transaction {i+1}/{num_transactions}: {transaction}")
            
            if queue_client.push_message(queue_file, transaction):
                success_count += 1
                
            # Add a small delay between requests to avoid file locking issues
            if i < num_transactions - 1:  # No need to delay after the last one
                time.sleep(0.1)  # 100ms delay between requests
        
        if success_count == 0:
            return jsonify({"success": False, "message": "Failed to push any transactions"})
        
        return jsonify({
            "success": True,
            "message": f"Successfully pushed {success_count} out of {num_transactions} transactions to {os.path.basename(queue_file)}."
        })
    except Exception as e:
        logger.error(f"Error in push_transactions: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/get_results')
def get_results():
    """Get prediction results from the results queue."""
    try:
        # Check if results queue exists
        if not queue_client.queue_exists(queue_client.results_queue_file):
            logger.warning(f"Results queue {CONFIG['results_queue_file']} doesn't exist")
            return jsonify({
                "success": True,
                "count": 0,
                "results": [],
                "message": f"Results queue {CONFIG['results_queue_file']} doesn't exist"
            })
        
        # Get results from queue without removing them
        results = queue_client.pull_messages(queue_client.results_queue_file, count=20, remove=False)
        
        # Extract content from message objects
        result_contents = [msg["content"] for msg in results]
        
        return jsonify({
            "success": True,
            "count": len(result_contents),
            "results": result_contents
        })
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/queue_status')
def queue_status():
    """Get the status of the queues."""
    try:
        # Check if queue files exist
        transaction_queue_exists = queue_client.queue_exists(queue_client.transaction_queue_file)
        transaction_queue2_exists = queue_client.queue_exists(queue_client.transaction_queue_file2)
        results_queue_exists = queue_client.queue_exists(queue_client.results_queue_file)
        
        # Get queue sizes
        transaction_queue_size = queue_client.get_queue_size(queue_client.transaction_queue_file)
        transaction_queue2_size = queue_client.get_queue_size(queue_client.transaction_queue_file2)
        results_queue_size = queue_client.get_queue_size(queue_client.results_queue_file)
        
        return jsonify({
            "success": True,
            "transaction_queue": {
                "exists": transaction_queue_exists,
                "size": transaction_queue_size,
                "name": CONFIG['transaction_queue_file']
            },
            "transaction_queue2": {
                "exists": transaction_queue2_exists,
                "size": transaction_queue2_size,
                "name": CONFIG['transaction_queue_file2']
            },
            "results_queue": {
                "exists": results_queue_exists,
                "size": results_queue_size,
                "name": CONFIG['results_queue_file']
            }
        })
    except Exception as e:
        logger.error(f"Error in queue_status: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/force_refresh')
def force_refresh():
    """Force refresh the queue status with caching disabled."""
    try:
        # Log queue status
        transaction_queue_exists = queue_client.queue_exists(queue_client.transaction_queue_file)
        transaction_queue2_exists = queue_client.queue_exists(queue_client.transaction_queue_file2)
        results_queue_exists = queue_client.queue_exists(queue_client.results_queue_file)
        
        transaction_queue_size = queue_client.get_queue_size(queue_client.transaction_queue_file)
        transaction_queue2_size = queue_client.get_queue_size(queue_client.transaction_queue_file2)
        results_queue_size = queue_client.get_queue_size(queue_client.results_queue_file)
        
        logger.info(f"Force refresh - TQ1 exists: {transaction_queue_exists}, size: {transaction_queue_size}")
        logger.info(f"Force refresh - TQ2 exists: {transaction_queue2_exists}, size: {transaction_queue2_size}")
        logger.info(f"Force refresh - PQ1 exists: {results_queue_exists}, size: {results_queue_size}")
        
        # Redirect back to main page with cache busting query parameter
        return redirect(f"/?refresh={int(time.time())}")
    except Exception as e:
        logger.error(f"Force refresh error: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template if it doesn't exist
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
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
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Queue Status</h5>
                    </div>
                    <div class="card-body">
                        <p>
                            <strong>Transaction Queue 1 (TQ1):</strong> 
                            <span class="status-indicator 
                                {% if transaction_queue_exists %}status-running
                                {% else %}status-not-running{% endif %}"></span>
                            {{ "Exists" if transaction_queue_exists else "Does Not Exist" }}
                            {% if transaction_queue_exists %}
                            ({{ transaction_queue_size }} messages)
                            {% endif %}
                        </p>
                        <p>
                            <strong>Transaction Queue 2 (TQ2):</strong> 
                            <span class="status-indicator 
                                {% if transaction_queue2_exists %}status-running
                                {% else %}status-not-running{% endif %}"></span>
                            {{ "Exists" if transaction_queue2_exists else "Does Not Exist" }}
                            {% if transaction_queue2_exists %}
                            ({{ transaction_queue2_size }} messages)
                            {% endif %}
                        </p>
                        <p>
                            <strong>Results Queue (PQ1):</strong> 
                            <span class="status-indicator 
                                {% if results_queue_exists %}status-running
                                {% else %}status-not-running{% endif %}"></span>
                            {{ "Exists" if results_queue_exists else "Does Not Exist" }}
                            {% if results_queue_exists %}
                            ({{ results_queue_size }} messages)
                            {% endif %}
                        </p>
                        
                        <a href="/force_refresh" class="btn btn-sm btn-secondary">Refresh Status</a>
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
                                <input type="number" class="form-control" id="numTransactions" name="num_transactions" value="5" min="1" max="100">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Target Queue</label>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="target_queue" id="tq1" value="TQ1" checked>
                                    <label class="form-check-label" for="tq1">
                                        Transaction Queue 1 (TQ1)
                                    </label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="target_queue" id="tq2" value="TQ2">
                                    <label class="form-check-label" for="tq2">
                                        Transaction Queue 2 (TQ2)
                                    </label>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary">
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
                        <button id="refreshResultsBtn" class="btn btn-sm btn-outline-primary">
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
                            <li>The system is configured to work with the following queue files:
                                <ul>
                                    <li><strong>Transaction Queue 1:</strong> {{ transaction_queue }}</li>
                                    <li><strong>Transaction Queue 2:</strong> {{ transaction_queue2 }}</li>
                                    <li><strong>Results Queue:</strong> {{ results_queue }}</li>
                                </ul>
                            </li>
                            <li>Push sample transactions to either TQ1 or TQ2</li>
                            <li>Start the MPI prediction service using one of these methods:
                                <ul>
                                    <li><code>mpirun -n 6 python prediction_service.py</code> (local mode)</li>
                                    <li><code>docker-compose up mpi_master</code> (Docker mode - if web UI is already running)</li>
                                    <li><code>docker-compose up</code> (Docker mode - to start everything including web UI)</li>
                                </ul>
                            </li>
                            <li>The prediction service will process transactions from the queue with the most messages</li>
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
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=7600, debug=False)
