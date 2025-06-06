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
            
            # Write back to queue file
            success = self._write_queue(queue_file, messages)
            if success:
                logger.info(f"Successfully pushed message to {os.path.basename(queue_file)}")
            else:
                logger.error(f"Failed to push message to {os.path.basename(queue_file)}")
            
            return success
        except Exception as e:
            logger.error(f"Error pushing message to {queue_file}: {str(e)}")
            return False
    
    def pull_messages(self, queue_file, count=1, remove=True):
        """Pull multiple messages from the specified queue file."""
        try:
            # Read all messages
            messages = self._read_queue(queue_file)
            
            if not messages:
                return []
            
            # Get the specified number of messages
            count = min(count, len(messages))
            pulled_messages = messages[:count]
            
            # Remove messages if specified
            if remove and pulled_messages:
                remaining_messages = messages[count:]
                self._write_queue(queue_file, remaining_messages)
                logger.info(f"Pulled and removed {len(pulled_messages)} messages from {os.path.basename(queue_file)}")
            else:
                logger.info(f"Pulled {len(pulled_messages)} messages from {os.path.basename(queue_file)} without removing")
            
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
    transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
    amount = random.uniform(100, 10000)
    transaction_count = random.randint(1, 50)
    time_since_last_transaction = random.randint(1, 1000)
    customer_risk_score = random.uniform(0, 1)  # Add customer risk score
    vendor_risk_score = random.uniform(0, 1)
    
    # Create a transaction with the required fields for the fraud detection model
    return {
        "transaction_id": transaction_id,
        "amount": round(amount, 2),
        "transaction_count": transaction_count,
        "time_since_last_transaction": time_since_last_transaction,
        "customer_risk_score": round(customer_risk_score, 2),  # Include customer risk score
        "vendor_risk_score": round(vendor_risk_score, 2)
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
                          transaction_queue_size=transaction_queue_size,
                          transaction_queue2_exists=transaction_queue2_exists,
                          transaction_queue2_size=transaction_queue2_size,
                          results_queue_exists=results_queue_exists,
                          results_queue_size=results_queue_size)

@app.route('/push_transactions', methods=['POST'])
def push_transactions():
    """Push sample transactions to the transaction queue."""
    try:
        # Get form data
        count = request.form.get('count', '1')
        try:
            count = int(count)
            if count < 1 or count > 10:
                return jsonify({"success": False, "message": "Count must be between 1 and 10"})
        except ValueError:
            return jsonify({"success": False, "message": "Count must be a number"})
        
        queue_choice = request.form.get('queue', 'TQ1')
        if queue_choice == 'TQ1':
            queue_file = queue_client.transaction_queue_file
        elif queue_choice == 'TQ2':
            queue_file = queue_client.transaction_queue_file2
        else:
            return jsonify({"success": False, "message": "Invalid queue selection"})
        
        # Generate and push transactions
        success_count = 0
        transactions = []
        for _ in range(count):
            transaction = generate_sample_transaction()
            transactions.append(transaction)
            if queue_client.push_message(queue_file, transaction):
                success_count += 1
        
        if success_count > 0:
            return jsonify({
                "success": True, 
                "message": f"Successfully pushed {success_count} transactions",
                "transactions": transactions
            })
        else:
            return jsonify({"success": False, "message": "Failed to push any transactions"})
    
    except Exception as e:
        logger.error(f"Error in push_transactions: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/get_results')
def get_results():
    """Get prediction results from the results queue."""
    try:
        # Check if results queue exists
        if not queue_client.queue_exists(queue_client.results_queue_file):
            logger.warning(f"Results queue file doesn't exist or is empty")
            return jsonify({
                "success": True,
                "count": 0,
                "results": [],
                "message": "No results available yet. Try pushing some transactions first."
            })
        
        # Get all results without removing them from the queue
        results = queue_client._read_queue(queue_client.results_queue_file)
        
        if not results:
            return jsonify({
                "success": True,
                "count": 0,
                "results": [],
                "message": "No prediction results found. The MPI service might still be processing."
            })
        
        # Sort results by timestamp if available
        try:
            results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        except Exception as e:
            logger.warning(f"Error sorting results: {str(e)}")
        
        # Process results to make them more user-friendly
        processed_results = []
        for result in results:
            # Extract the content if it's nested
            if 'content' in result:
                result_content = result['content']
            else:
                result_content = result
            
            # Handle potential type conversion issues
            try:
                fraud_probability = float(result_content.get('fraud_probability', 0))
                fraud_probability_pct = round(fraud_probability * 100, 2)
            except (ValueError, TypeError):
                fraud_probability_pct = 0
                
            # Add formatted fields for better display
            processed_result = {
                'transaction_id': result_content.get('transaction_id', 'Unknown'),
                'is_fraud': result_content.get('is_fraud', False),
                'fraud_probability': fraud_probability_pct,
                'fraud_probability_display': f"{fraud_probability_pct}%",
                'timestamp': result.get('timestamp', 'Unknown time'),
                'details': result_content
            }
            processed_results.append(processed_result)
        
        # Limit to the latest 20 results to avoid overwhelming the UI
        processed_results = processed_results[:20]
        
        return jsonify({
            "success": True,
            "count": len(processed_results),
            "results": processed_results
        })
    
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "count": 0,
            "results": []
        })

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
                "file": os.path.basename(queue_client.transaction_queue_file)
            },
            "transaction_queue2": {
                "exists": transaction_queue2_exists,
                "size": transaction_queue2_size,
                "file": os.path.basename(queue_client.transaction_queue_file2)
            },
            "results_queue": {
                "exists": results_queue_exists,
                "size": results_queue_size,
                "file": os.path.basename(queue_client.results_queue_file)
            }
        })
    except Exception as e:
        logger.error(f"Error in queue_status: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/force_refresh')
def force_refresh():
    """Force refresh the queue status with caching disabled."""
    return redirect(url_for('index', _=int(time.time())))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template if it doesn't exist
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MPI Fraud Detection Service</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .json-display {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
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
        .fraud-alert {
            color: #dc3545;
            font-weight: bold;
        }
        .fraud-safe {
            color: #28a745;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">MPI Fraud Detection Service</h1>
        
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
                                {% else %}status-not-running{% endif %}">
                            </span>
                            {% if transaction_queue_exists %}
                            Active ({{ transaction_queue_size }} messages)
                            {% else %}
                            Not active
                            {% endif %}
                        </p>
                        
                        <p>
                            <strong>Transaction Queue 2 (TQ2):</strong> 
                            <span class="status-indicator 
                                {% if transaction_queue2_exists %}status-running
                                {% else %}status-not-running{% endif %}">
                            </span>
                            {% if transaction_queue2_exists %}
                            Active ({{ transaction_queue2_size }} messages)
                            {% else %}
                            Not active
                            {% endif %}
                        </p>
                        
                        <p>
                            <strong>Results Queue (PQ1):</strong> 
                            <span class="status-indicator 
                                {% if results_queue_exists %}status-running
                                {% else %}status-not-running{% endif %}">
                            </span>
                            {% if results_queue_exists %}
                            Active ({{ results_queue_size }} messages)
                            {% else %}
                            Not active
                            {% endif %}
                        </p>
                        
                        <a href="/force_refresh" class="btn btn-sm btn-secondary">Refresh Status</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>MPI Service Instructions</h5>
                    </div>
                    <div class="card-body">
                        <p>This web UI allows you to interact with the MPI-based fraud detection service:</p>
                        <ol>
                            <li>Use the <strong>Push Transactions</strong> form to send sample transactions to the queue</li>
                            <li>The MPI service will process these transactions using multiple processors</li>
                            <li>View the prediction results in the <strong>Prediction Results</strong> section</li>
                        </ol>
                        <p class="text-info">The MPI service reads transactions from the queue files, processes them in parallel, and writes results back to the results queue.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Push Transactions</h5>
                    </div>
                    <div class="card-body">
                        <form id="pushTransactionsForm">
                            <div class="mb-3">
                                <label for="count" class="form-label">Number of Transactions</label>
                                <input type="number" class="form-control" id="count" name="count" min="1" max="10" value="1">
                                <div class="form-text">How many random transactions to generate (1-10)</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="queue" class="form-label">Target Queue</label>
                                <select class="form-select" id="queue" name="queue">
                                    <option value="TQ1">Transaction Queue 1 (TQ1)</option>
                                    <option value="TQ2">Transaction Queue 2 (TQ2)</option>
                                </select>
                            </div>
                            
                            <button type="submit" class="btn btn-primary" id="submitBtn">Push Transactions</button>
                        </form>
                        
                        <div id="transactionResult" class="mt-3" style="display: none;">
                            <h6>Pushed Transactions:</h6>
                            <div id="transactionResultContent" class="json-display"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Prediction Results</h5>
                    </div>
                    <div class="card-body">
                        <button id="refreshResultsBtn" class="btn btn-primary mb-3">Refresh Results</button>
                        
                        <div id="resultsContainer">
                            <p class="text-muted">Click "Refresh Results" to check for new prediction results.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Push Transactions Form
            const pushTransactionsForm = document.getElementById('pushTransactionsForm');
            const submitBtn = document.getElementById('submitBtn');
            const transactionResult = document.getElementById('transactionResult');
            const transactionResultContent = document.getElementById('transactionResultContent');
            
            if (pushTransactionsForm) {
                pushTransactionsForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
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
                            // Show the pushed transactions
                            transactionResult.style.display = 'block';
                            transactionResultContent.textContent = JSON.stringify(data.transactions, null, 2);
                            
                            // Refresh the queue status
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
                                if (data.message) {
                                    resultsContainer.innerHTML = `<p class="text-muted">${data.message}</p>`;
                                } else {
                                    resultsContainer.innerHTML = '<p class="text-muted">No results found. The prediction service might still be processing or no transactions have been processed yet.</p>';
                                }
                            } else {
                                let html = `<p>Found ${data.count} prediction results:</p>`;
                                
                                data.results.forEach((result, index) => {
                                    const fraudClass = result.is_fraud ? 'fraud-alert' : 'fraud-safe';
                                    const fraudText = result.is_fraud ? 'FRAUD DETECTED' : 'No fraud detected';
                                    
                                    html += `
                                    <div class="card mb-3">
                                        <div class="card-header d-flex justify-content-between align-items-center">
                                            <h6 class="mb-0">Transaction: ${result.transaction_id}</h6>
                                            <span class="${fraudClass}">${fraudText}</span>
                                        </div>
                                        <div class="card-body">
                                            <p><strong>Fraud Probability:</strong> ${result.fraud_probability_display}</p>
                                            <p><strong>Timestamp:</strong> ${result.timestamp}</p>
                                            <p><strong>Details:</strong></p>
                                            <div class="json-display">${JSON.stringify(result.details, null, 2)}</div>
                                        </div>
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
