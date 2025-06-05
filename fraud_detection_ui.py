#!/usr/bin/env python
# encoding: utf8

import os
import sys
import time
import json
import random
import argparse
import traceback
import webbrowser
import threading
import requests
from datetime import datetime, timezone
from flask import Flask, request, render_template, jsonify

# Constants
QUEUE_SERVICE_URL = 'http://localhost:8000'
TRANSACTION_QUEUE = 'TQ1'  # Transaction Queue name from Assignment 3
PREDICTION_QUEUE = 'PQ1'    # Prediction Queue name from Assignment 3
MAX_RETRIES = 5  # Maximum number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds

# Initialize Flask app
app = Flask(__name__)

# In-memory storage for transactions and predictions
transactions = []
predictions = []
queue_service_status = {'healthy': False, 'last_checked': None}


def check_queue_service_health():
    """Check if the queue service is healthy"""
    try:
        response = requests.get(f"{QUEUE_SERVICE_URL}/health", timeout=5)
        queue_service_status['healthy'] = response.status_code == 200
        queue_service_status['last_checked'] = datetime.now(timezone.UTC).isoformat()
        return queue_service_status['healthy']
    except requests.exceptions.RequestException:
        queue_service_status['healthy'] = False
        queue_service_status['last_checked'] = datetime.now(timezone.UTC).isoformat()
        return False
    except Exception as e:
        print(f"Unexpected error checking queue service health: {e}")
        queue_service_status['healthy'] = False
        queue_service_status['last_checked'] = datetime.now(timezone.UTC).isoformat()
        return False


def push_to_transaction_queue(transaction):
    """Push a transaction to the queue service with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            # In a real implementation, this would use proper authentication
            headers = {"Authorization": "Bearer mock_token", "Content-Type": "application/json"}
            
            message = {
                "content": transaction,
                "message_type": "transaction"
            }
            
            response = requests.post(
                f"{QUEUE_SERVICE_URL}/api/queues/{TRANSACTION_QUEUE}/push",
                headers=headers,
                json=message,
                timeout=5  # Add timeout to prevent hanging
            )
            
            if response.status_code == 201:
                print(f"Transaction sent successfully: {transaction['transaction_id']}")
                return True
            else:
                print(f"Failed to send transaction: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Giving up.")
                    return False
        except requests.exceptions.RequestException as e:
            print(f"Error sending transaction (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Giving up.")
                return False
        except Exception as e:
            print(f"Unexpected error sending transaction: {e}")
            traceback.print_exc()
            return False
    
    return False


def fetch_from_prediction_queue():
    """Fetch predictions from the queue service with retry logic"""
    results = []
    
    for attempt in range(MAX_RETRIES):
        try:
            # In a real implementation, this would use proper authentication
            headers = {"Authorization": "Bearer mock_token"}
            
            # Try to pull multiple predictions (up to 10)
            for _ in range(10):
                response = requests.post(
                    f"{QUEUE_SERVICE_URL}/api/queues/{PREDICTION_QUEUE}/pull",
                    headers=headers,
                    timeout=5  # Add timeout to prevent hanging
                )
                
                if response.status_code == 200:
                    message = response.json()
                    if message and "content" in message:
                        results.append(message["content"])
                elif response.status_code == 404:
                    # Queue is empty
                    break
                else:
                    print(f"Unexpected status code: {response.status_code} - {response.text}")
                    break
            
            # If we got here without exception, break the retry loop
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching predictions (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Giving up.")
                # If all retries fail, generate mock predictions for any transactions without predictions
                return generate_mock_predictions()
        except Exception as e:
            print(f"Unexpected error fetching predictions: {e}")
            traceback.print_exc()
            # If exception occurs, generate mock predictions
            return generate_mock_predictions()
    
    # If no predictions were fetched and all retries failed, generate mock predictions
    if not results:
        return generate_mock_predictions()
    
    return results


def generate_mock_predictions():
    """Generate mock predictions for transactions that don't have predictions yet"""
    mock_results = []
    
    # Find transactions without predictions
    prediction_tx_ids = {p["transaction_id"] for p in predictions}
    unpredicted_txs = [tx for tx in transactions if tx["transaction_id"] not in prediction_tx_ids]
    
    for tx in unpredicted_txs:
        # Generate a mock prediction
        is_fraud = random.random() < 0.2  # 20% chance of fraud
        confidence = random.uniform(0.6, 0.99) if is_fraud else random.uniform(0.7, 0.99)
        
        prediction = {
            "transaction_id": tx["transaction_id"],
            "prediction": is_fraud,
            "confidence": confidence,
            "model_version": "mock-1.0",
            "timestamp": datetime.now(timezone.UTC).isoformat(),
            "processor_rank": -1,  # Indicates mock prediction
            "mock": True
        }
        
        mock_results.append(prediction)
    
    return mock_results


@app.route('/')
def index():
    """Render the main page"""
    # Check queue service health
    check_queue_service_health()
    return render_template('index.html', 
                           transactions=transactions, 
                           predictions=predictions,
                           queue_status=queue_service_status)


@app.route('/submit', methods=['POST'])
def submit_transaction():
    """Submit a transaction"""
    try:
        # Get form data
        customer_id = request.form.get('customer_id', f"cust-{random.randint(1000, 9999)}")
        amount = float(request.form.get('amount', 0))
        vendor_id = int(request.form.get('vendor_id', 0))
        status = request.form.get('status', 'submitted')
        
        # Create transaction
        transaction = {
            "transaction_id": f"tx-{int(time.time())}-{random.randint(1000, 9999)}",
            "customer_id": customer_id,
            "amount": amount,
            "vendor_id": vendor_id,
            "status": status,
            "timestamp": datetime.now(timezone.UTC).isoformat()
        }
        
        # Add to in-memory storage
        transactions.append(transaction)
        
        # Check queue service health
        is_healthy = check_queue_service_health()
        
        # Push to queue service if healthy
        queue_success = False
        if is_healthy:
            queue_success = push_to_transaction_queue(transaction)
        
        return jsonify({
            "success": True,
            "transaction": transaction,
            "queue_service": {
                "healthy": is_healthy,
                "success": queue_success
            }
        })
    except Exception as e:
        print(f"Error submitting transaction: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/generate_random', methods=['POST'])
def generate_random():
    """Generate random transactions"""
    try:
        count = int(request.form.get('count', 1))
        count = min(count, 10)  # Limit to 10 transactions at a time
        
        new_transactions = []
        for _ in range(count):
            # Create transaction
            transaction = {
                "transaction_id": f"tx-{int(time.time())}-{random.randint(1000, 9999)}",
                "customer_id": f"cust-{random.randint(1000, 9999)}",
                "amount": round(random.uniform(10.0, 1000.0), 2),
                "vendor_id": random.randint(1, 100),
                "status": random.choice(['submitted', 'accepted', 'rejected']),
                "timestamp": datetime.now(timezone.UTC).isoformat()
            }
            
            # Add to local storage
            transactions.append(transaction)
            new_transactions.append(transaction)
            
            # Push to queue service
            push_to_transaction_queue(transaction)
            
            # Small delay to ensure unique timestamps
            time.sleep(0.01)
        
        return jsonify({
            "success": True,
            "transactions": new_transactions,
            "queue_status": queue_service_status
        })
    except Exception as e:
        print(f"Error generating random transactions: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/check_predictions', methods=['POST'])
def check_predictions():
    """Check for new predictions"""
    try:
        # Check queue service health
        check_queue_service_health()
        
        # Fetch predictions
        new_predictions = fetch_from_prediction_queue()
        
        # Add to local storage (avoid duplicates)
        prediction_ids = {p["transaction_id"] for p in predictions}
        for prediction in new_predictions:
            if prediction["transaction_id"] not in prediction_ids:
                predictions.append(prediction)
                prediction_ids.add(prediction["transaction_id"])
        
        return jsonify({
            "success": True,
            "new_predictions": new_predictions,
            "total_predictions": len(predictions),
            "queue_status": queue_service_status
        })
    except Exception as e:
        print(f"Error checking predictions: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_data():
    """Clear all local data"""
    try:
        global transactions, predictions
        transactions = []
        predictions = []
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error clearing data: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


def create_templates_folder():
    """Create the templates folder and index.html file"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection UI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card-header {
            font-weight: bold;
            background-color: #f1f1f1;
        }
        .transaction-item, .prediction-item {
            border-bottom: 1px solid #eee;
            padding: 10px;
        }
        .transaction-item:last-child, .prediction-item:last-child {
            border-bottom: none;
        }
        .fraud-true {
            color: #dc3545;
            font-weight: bold;
        }
        .fraud-false {
            color: #28a745;
        }
        .mock-prediction {
            font-style: italic;
            opacity: 0.8;
        }
        .queue-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 14px;
            margin-left: 10px;
        }
        .queue-healthy {
            background-color: #d4edda;
            color: #155724;
        }
        .queue-unhealthy {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Fraud Detection System
            <span id="queue-status-badge" class="queue-status queue-unhealthy">Queue Service: Offline</span>
        </h1>
        
        <div class="row">
            <!-- Transaction Form -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Submit Transaction</div>
                    <div class="card-body">
                        <form id="transaction-form">
                            <div class="mb-3">
                                <label for="customer-id" class="form-label">Customer ID</label>
                                <input type="text" class="form-control" id="customer-id" placeholder="e.g., cust-1234">
                            </div>
                            <div class="mb-3">
                                <label for="amount" class="form-label">Amount</label>
                                <input type="number" class="form-control" id="amount" step="0.01" min="0" placeholder="e.g., 100.00">
                            </div>
                            <div class="mb-3">
                                <label for="vendor-id" class="form-label">Vendor ID</label>
                                <input type="number" class="form-control" id="vendor-id" placeholder="e.g., 42">
                            </div>
                            <div class="mb-3">
                                <label for="status" class="form-label">Status</label>
                                <select class="form-select" id="status">
                                    <option value="submitted">Submitted</option>
                                    <option value="accepted">Accepted</option>
                                    <option value="rejected">Rejected</option>
                                </select>
                            </div>
                            <button type="submit" class="btn btn-primary">Submit</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">Generate Random Transactions</div>
                    <div class="card-body">
                        <form id="random-form">
                            <div class="mb-3">
                                <label for="count" class="form-label">Number of Transactions</label>
                                <input type="number" class="form-control" id="count" min="1" max="10" value="3">
                            </div>
                            <button type="submit" class="btn btn-success">Generate</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">Controls</div>
                    <div class="card-body">
                        <button id="check-predictions-btn" class="btn btn-info me-2">Check for Predictions</button>
                        <button id="clear-data-btn" class="btn btn-danger">Clear All Data</button>
                    </div>
                </div>
            </div>
            
            <!-- Data Display -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        Transactions <span id="transaction-count" class="badge bg-primary">{{ transactions|length }}</span>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="transactions-container">
                            {% for tx in transactions %}
                            <div class="transaction-item" data-id="{{ tx.transaction_id }}">
                                <div><strong>ID:</strong> {{ tx.transaction_id }}</div>
                                <div><strong>Customer:</strong> {{ tx.customer_id }}</div>
                                <div><strong>Amount:</strong> ${{ tx.amount }}</div>
                                <div><strong>Status:</strong> {{ tx.status }}</div>
                                <div><strong>Vendor:</strong> {{ tx.vendor_id }}</div>
                            </div>
                            {% else %}
                            <div id="no-transactions">No transactions yet.</div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        Predictions <span id="prediction-count" class="badge bg-primary">{{ predictions|length }}</span>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="predictions-container">
                            {% for pred in predictions %}
                            <div class="prediction-item{% if pred.mock %} mock-prediction{% endif %}" data-id="{{ pred.transaction_id }}">
                                <div><strong>Transaction ID:</strong> {{ pred.transaction_id }}</div>
                                <div>
                                    <strong>Fraud:</strong> 
                                    <span class="{% if pred.prediction %}fraud-true{% else %}fraud-false{% endif %}">
                                        {{ pred.prediction }}
                                    </span>
                                </div>
                                <div><strong>Confidence:</strong> {{ "%.2f"|format(pred.confidence*100) }}%</div>
                                <div><strong>Model:</strong> {{ pred.model_version }}</div>
                                <div><strong>Processor:</strong> {{ pred.processor_rank }}</div>
                                {% if pred.mock %}<div><em>(Mock Prediction)</em></div>{% endif %}
                            </div>
                            {% else %}
                            <div id="no-predictions">No predictions yet.</div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update queue status badge
        function updateQueueStatus(healthy) {
            const badge = document.getElementById('queue-status-badge');
            if (healthy) {
                badge.className = 'queue-status queue-healthy';
                badge.textContent = 'Queue Service: Online';
            } else {
                badge.className = 'queue-status queue-unhealthy';
                badge.textContent = 'Queue Service: Offline';
            }
        }
        
        // Initialize queue status
        updateQueueStatus({{ queue_status.healthy|tojson }});
        
        // Add a transaction to the UI
        function addTransaction(transaction) {
            const container = document.getElementById('transactions-container');
            const noTransactions = document.getElementById('no-transactions');
            if (noTransactions) {
                noTransactions.remove();
            }
            
            const txElement = document.createElement('div');
            txElement.className = 'transaction-item';
            txElement.dataset.id = transaction.transaction_id;
            txElement.innerHTML = `
                <div><strong>ID:</strong> ${transaction.transaction_id}</div>
                <div><strong>Customer:</strong> ${transaction.customer_id}</div>
                <div><strong>Amount:</strong> $${transaction.amount}</div>
                <div><strong>Status:</strong> ${transaction.status}</div>
                <div><strong>Vendor:</strong> ${transaction.vendor_id}</div>
            `;
            
            container.appendChild(txElement);
            
            // Update count
            const countElement = document.getElementById('transaction-count');
            countElement.textContent = container.querySelectorAll('.transaction-item').length;
        }
        
        // Add a prediction to the UI
        function addPrediction(prediction) {
            const container = document.getElementById('predictions-container');
            const noPredictions = document.getElementById('no-predictions');
            if (noPredictions) {
                noPredictions.remove();
            }
            
            // Check if prediction already exists
            const existingPrediction = document.querySelector(`.prediction-item[data-id="${prediction.transaction_id}"]`);
            if (existingPrediction) {
                return; // Skip if already exists
            }
            
            const predElement = document.createElement('div');
            predElement.className = 'prediction-item';
            if (prediction.mock) {
                predElement.className += ' mock-prediction';
            }
            predElement.dataset.id = prediction.transaction_id;
            
            const fraudClass = prediction.prediction ? 'fraud-true' : 'fraud-false';
            const confidencePercent = (prediction.confidence * 100).toFixed(2);
            
            predElement.innerHTML = `
                <div><strong>Transaction ID:</strong> ${prediction.transaction_id}</div>
                <div>
                    <strong>Fraud:</strong> 
                    <span class="${fraudClass}">
                        ${prediction.prediction}
                    </span>
                </div>
                <div><strong>Confidence:</strong> ${confidencePercent}%</div>
                <div><strong>Model:</strong> ${prediction.model_version}</div>
                <div><strong>Processor:</strong> ${prediction.processor_rank}</div>
                ${prediction.mock ? '<div><em>(Mock Prediction)</em></div>' : ''}
            `;
            
            container.appendChild(predElement);
            
            // Update count
            const countElement = document.getElementById('prediction-count');
            countElement.textContent = container.querySelectorAll('.prediction-item').length;
        }
        
        // Submit transaction form
        document.getElementById('transaction-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('customer_id', document.getElementById('customer-id').value || `cust-${Math.floor(Math.random() * 9000) + 1000}`);
            formData.append('amount', document.getElementById('amount').value || Math.random() * 1000);
            formData.append('vendor_id', document.getElementById('vendor-id').value || Math.floor(Math.random() * 100) + 1);
            formData.append('status', document.getElementById('status').value);
            
            try {
                const response = await fetch('/submit', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    addTransaction(data.transaction);
                    updateQueueStatus(data.queue_status.healthy);
                    // Reset form
                    document.getElementById('transaction-form').reset();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
        
        // Generate random transactions
        document.getElementById('random-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('count', document.getElementById('count').value || 1);
            
            try {
                const response = await fetch('/generate_random', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    data.transactions.forEach(tx => addTransaction(tx));
                    updateQueueStatus(data.queue_status.healthy);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
        
        // Check for predictions
        document.getElementById('check-predictions-btn').addEventListener('click', async () => {
            try {
                const response = await fetch('/check_predictions', {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.success) {
                    data.new_predictions.forEach(pred => addPrediction(pred));
                    updateQueueStatus(data.queue_status.healthy);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
        
        // Clear all data
        document.getElementById('clear-data-btn').addEventListener('click', async () => {
            if (!confirm('Are you sure you want to clear all data?')) {
                return;
            }
            
            try {
                const response = await fetch('/clear', {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.success) {
                    // Clear UI
                    document.getElementById('transactions-container').innerHTML = '<div id="no-transactions">No transactions yet.</div>';
                    document.getElementById('predictions-container').innerHTML = '<div id="no-predictions">No predictions yet.</div>';
                    document.getElementById('transaction-count').textContent = '0';
                    document.getElementById('prediction-count').textContent = '0';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
        
        // Auto-check for predictions every 5 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/check_predictions', {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.success) {
                    data.new_predictions.forEach(pred => addPrediction(pred));
                    updateQueueStatus(data.queue_status.healthy);
                }
            } catch (error) {
                console.error('Error auto-checking predictions:', error);
            }
        }, 5000);
    </script>
</body>
</html>
'''
    
    index_path = os.path.join(templates_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"Created templates folder and index.html at {index_path}")


def open_browser(port):
    """Open the browser after a short delay"""
    def _open_browser():
        # Add a longer delay to ensure Flask server is fully started
        time.sleep(3)
        try:
            webbrowser.open(f'http://localhost:{port}/')
            print(f"Opened browser at http://localhost:{port}/")
        except Exception as e:
            print(f"Error opening browser: {e}")
    
    # Only open browser in the main process when not in debug mode or when in the main thread of debug mode
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        thread = threading.Thread(target=_open_browser)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the UI on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--queue-url", type=str, help="URL of the queue service")
    parser.add_argument("--no-debug", action="store_true", help="Disable Flask debug mode")
    
    args = parser.parse_args()
    
    # Override queue service URL if provided
    if args.queue_url:
        QUEUE_SERVICE_URL = args.queue_url
        print(f"Using queue service at: {QUEUE_SERVICE_URL}")
    
    # Create templates folder and index.html
    create_templates_folder()
    
    # Open browser
    if not args.no_browser and not os.environ.get('WERKZEUG_RUN_MAIN'):
        open_browser(args.port)
    
    # Run the app
    debug_mode = not args.no_debug
    app.run(
        host='127.0.0.1',
        port=args.port,
        debug=debug_mode,
        use_reloader=debug_mode
    )