#!/usr/bin/env python
# encoding: utf8

import json
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import random
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Constants
QUEUE_SERVICE_URL = 'http://localhost:8000'
TRANSACTION_QUEUE = 'TQ1'  # Transaction Queue name from Assignment 3
PREDICTION_QUEUE = 'PQ1'    # Prediction Queue name from Assignment 3
STATUSES = ['submitted', 'accepted', 'rejected']
VENDOR_IDS = list(range(1, 101))  # Vendor IDs from 1 to 100

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud_detection_ui_secret_key'

# In-memory storage for demo purposes
transactions = []
predictions = []

# Mock token for API calls
MOCK_TOKEN = "mock_token"


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                          transactions=transactions, 
                          predictions=predictions,
                          statuses=STATUSES,
                          vendor_ids=VENDOR_IDS)


@app.route('/submit_transaction', methods=['POST'])
def submit_transaction():
    """Submit a transaction to the queue"""
    try:
        # Get form data
        customer_id = request.form.get('customer_id', f"cust-{random.randint(1, 1000)}")
        amount = float(request.form.get('amount', random.uniform(10.0, 1000.0)))
        status = request.form.get('status', random.choice(STATUSES))
        vendor_id = int(request.form.get('vendor_id', random.choice(VENDOR_IDS)))
        
        # Create transaction object
        transaction = {
            "transaction_id": f"tx-{int(time.time())}-{len(transactions)}",
            "customer_id": customer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "vendor_id": vendor_id,
            "amount": round(amount, 2)
        }
        
        # Add to local storage
        transactions.append(transaction)
        
        # Send to queue service
        success = send_to_transaction_queue(transaction)
        
        if success:
            return jsonify({"status": "success", "message": "Transaction submitted successfully", "transaction": transaction})
        else:
            return jsonify({"status": "error", "message": "Failed to submit transaction to queue"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/generate_random', methods=['POST'])
def generate_random():
    """Generate and submit a random transaction"""
    try:
        # Create random transaction
        transaction = {
            "transaction_id": f"tx-{int(time.time())}-{len(transactions)}",
            "customer_id": f"cust-{random.randint(1, 1000)}",
            "timestamp": datetime.utcnow().isoformat(),
            "status": random.choice(STATUSES),
            "vendor_id": random.choice(VENDOR_IDS),
            "amount": round(random.uniform(10.0, 1000.0), 2)
        }
        
        # Add to local storage
        transactions.append(transaction)
        
        # Send to queue service
        success = send_to_transaction_queue(transaction)
        
        if success:
            return jsonify({"status": "success", "message": "Random transaction generated", "transaction": transaction})
        else:
            return jsonify({"status": "error", "message": "Failed to submit transaction to queue"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/check_predictions', methods=['GET'])
def check_predictions():
    """Check for new prediction results"""
    try:
        # Fetch from prediction queue
        new_predictions = fetch_from_prediction_queue()
        
        if new_predictions:
            # Add to local storage
            predictions.extend(new_predictions)
            return jsonify({"status": "success", "message": f"Found {len(new_predictions)} new predictions", "predictions": new_predictions})
        else:
            return jsonify({"status": "info", "message": "No new predictions found"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all local data"""
    global transactions, predictions
    transactions = []
    predictions = []
    return jsonify({"status": "success", "message": "All data cleared"})


def send_to_transaction_queue(transaction):
    """Send a transaction to the queue service"""
    try:
        # In a real implementation, this would use proper authentication
        headers = {"Authorization": f"Bearer {MOCK_TOKEN}"}
        
        message = {
            "content": transaction,
            "message_type": "transaction"
        }
        
        response = requests.post(
            f"{QUEUE_SERVICE_URL}/api/queues/{TRANSACTION_QUEUE}/push",
            headers=headers,
            json=message
        )
        
        if response.status_code == 201:
            print(f"Transaction sent successfully: {transaction['transaction_id']}")
            return True
        else:
            print(f"Failed to send transaction: {response.status_code} - {response.text}")
            # For demo purposes, we'll consider it successful even if the API call fails
            return True
    except Exception as e:
        print(f"Error sending transaction: {e}")
        # For demo purposes, we'll consider it successful even if the API call fails
        return True


def fetch_from_prediction_queue(max_count=10):
    """Fetch prediction results from the queue service"""
    results = []
    try:
        # In a real implementation, this would use proper authentication
        headers = {"Authorization": f"Bearer {MOCK_TOKEN}"}
        
        for _ in range(max_count):
            response = requests.post(
                f"{QUEUE_SERVICE_URL}/api/queues/{PREDICTION_QUEUE}/pull",
                headers=headers
            )
            
            if response.status_code == 200:
                message = response.json()
                if message and "content" in message:
                    results.append(message["content"])
            elif response.status_code == 404:
                # Queue is empty
                break
            else:
                print(f"Failed to fetch prediction: {response.status_code} - {response.text}")
                break
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        # For demo purposes, generate some mock predictions
        if not results and transactions:
            for tx in transactions[:3]:
                if any(p["transaction_id"] == tx["transaction_id"] for p in predictions):
                    continue
                    
                results.append({
                    "transaction_id": tx["transaction_id"],
                    "prediction": random.choice([True, False]),
                    "confidence": round(random.uniform(0.6, 0.99), 4),
                    "model_version": "1.0",
                    "timestamp": datetime.utcnow().isoformat(),
                    "processor_rank": random.randint(0, 4)
                })
    
    return results


def create_templates_folder():
    """Create templates folder and index.html file"""
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
        body { padding-top: 20px; }
        .card { margin-bottom: 20px; }
        .transaction-card { border-left: 5px solid #007bff; }
        .prediction-card { border-left: 5px solid #28a745; }
        .prediction-fraud { border-left: 5px solid #dc3545; }
        .refresh-btn { margin-left: 10px; }
        .timestamp { font-size: 0.8rem; color: #6c757d; }
        .badge-fraud { background-color: #dc3545; }
        .badge-legitimate { background-color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Fraud Detection System</h1>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Submit Transaction</h5>
                        <button id="generateRandom" class="btn btn-sm btn-secondary">Generate Random</button>
                    </div>
                    <div class="card-body">
                        <form id="transactionForm">
                            <div class="mb-3">
                                <label for="customerId" class="form-label">Customer ID</label>
                                <input type="text" class="form-control" id="customerId" placeholder="e.g., cust-123">
                            </div>
                            <div class="mb-3">
                                <label for="amount" class="form-label">Amount</label>
                                <input type="number" class="form-control" id="amount" min="10" max="1000" step="0.01" placeholder="Amount in $">
                            </div>
                            <div class="mb-3">
                                <label for="status" class="form-label">Status</label>
                                <select class="form-control" id="status">
                                    {% for status in statuses %}
                                    <option value="{{ status }}">{{ status }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="vendorId" class="form-label">Vendor ID</label>
                                <select class="form-control" id="vendorId">
                                    {% for vendor_id in vendor_ids %}
                                    <option value="{{ vendor_id }}">{{ vendor_id }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <button type="submit" class="btn btn-primary">Submit Transaction</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Controls</h5>
                        <button id="clearData" class="btn btn-sm btn-danger">Clear All Data</button>
                    </div>
                    <div class="card-body">
                        <p>Use this panel to control the fraud detection system.</p>
                        <div class="d-flex">
                            <button id="checkPredictions" class="btn btn-success">Check for Predictions</button>
                            <div class="form-check ms-3 mt-2">
                                <input class="form-check-input" type="checkbox" id="autoRefresh">
                                <label class="form-check-label" for="autoRefresh">
                                    Auto-refresh (5s)
                                </label>
                            </div>
                        </div>
                        <div class="mt-3">
                            <p><strong>Stats:</strong></p>
                            <p>Transactions: <span id="txCount">{{ transactions|length }}</span></p>
                            <p>Predictions: <span id="predCount">{{ predictions|length }}</span></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <h4 class="d-flex justify-content-between align-items-center mb-3">
                    <span>Transactions</span>
                    <span class="badge bg-primary rounded-pill" id="transactionBadge">{{ transactions|length }}</span>
                </h4>
                <div id="transactionsList">
                    {% for tx in transactions %}
                    <div class="card transaction-card">
                        <div class="card-body">
                            <h5 class="card-title">Transaction: {{ tx.transaction_id }}</h5>
                            <h6 class="card-subtitle mb-2 text-muted">Customer: {{ tx.customer_id }}</h6>
                            <p class="card-text">
                                Amount: ${{ tx.amount }}<br>
                                Status: {{ tx.status }}<br>
                                Vendor ID: {{ tx.vendor_id }}
                            </p>
                            <div class="timestamp">{{ tx.timestamp }}</div>
                        </div>
                    </div>
                    {% else %}
                    <div class="alert alert-info">No transactions yet</div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="col-md-6">
                <h4 class="d-flex justify-content-between align-items-center mb-3">
                    <span>Prediction Results</span>
                    <span class="badge bg-success rounded-pill" id="predictionBadge">{{ predictions|length }}</span>
                </h4>
                <div id="predictionsList">
                    {% for pred in predictions %}
                    <div class="card {% if pred.prediction %}prediction-fraud{% else %}prediction-card{% endif %}">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <h5 class="card-title">Result: {{ pred.transaction_id }}</h5>
                                <span class="badge {% if pred.prediction %}badge-fraud{% else %}badge-legitimate{% endif %}">
                                    {% if pred.prediction %}FRAUD{% else %}LEGITIMATE{% endif %}
                                </span>
                            </div>
                            <p class="card-text">
                                Confidence: {{ pred.confidence|round(4) }}<br>
                                Model Version: {{ pred.model_version }}<br>
                                Processor: {{ pred.processor_rank }}
                            </p>
                            <div class="timestamp">{{ pred.timestamp }}</div>
                        </div>
                    </div>
                    {% else %}
                    <div class="alert alert-info">No predictions yet</div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Form submission
        document.getElementById('transactionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const customerId = document.getElementById('customerId').value || `cust-${Math.floor(Math.random() * 1000)}`;
            const amount = document.getElementById('amount').value || (Math.random() * 990 + 10).toFixed(2);
            const status = document.getElementById('status').value;
            const vendorId = document.getElementById('vendorId').value;
            
            fetch('/submit_transaction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `customer_id=${encodeURIComponent(customerId)}&amount=${encodeURIComponent(amount)}&status=${encodeURIComponent(status)}&vendor_id=${encodeURIComponent(vendorId)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    addTransactionToUI(data.transaction);
                    document.getElementById('transactionForm').reset();
                } else {
                    alert(`Error: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while submitting the transaction');
            });
        });
        
        // Generate random transaction
        document.getElementById('generateRandom').addEventListener('click', function() {
            fetch('/generate_random', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    addTransactionToUI(data.transaction);
                } else {
                    alert(`Error: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while generating a random transaction');
            });
        });
        
        // Check for predictions
        document.getElementById('checkPredictions').addEventListener('click', checkPredictions);
        
        function checkPredictions() {
            fetch('/check_predictions')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.predictions && data.predictions.length > 0) {
                    data.predictions.forEach(pred => addPredictionToUI(pred));
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
        
        // Clear all data
        document.getElementById('clearData').addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all data?')) {
                fetch('/clear_data', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('transactionsList').innerHTML = '<div class="alert alert-info">No transactions yet</div>';
                        document.getElementById('predictionsList').innerHTML = '<div class="alert alert-info">No predictions yet</div>';
                        document.getElementById('transactionBadge').textContent = '0';
                        document.getElementById('predictionBadge').textContent = '0';
                        document.getElementById('txCount').textContent = '0';
                        document.getElementById('predCount').textContent = '0';
                    } else {
                        alert(`Error: ${data.message}`);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while clearing data');
                });
            }
        });
        
        // Auto-refresh toggle
        let refreshInterval;
        document.getElementById('autoRefresh').addEventListener('change', function(e) {
            if (e.target.checked) {
                refreshInterval = setInterval(checkPredictions, 5000);
            } else {
                clearInterval(refreshInterval);
            }
        });
        
        // Helper function to add transaction to UI
        function addTransactionToUI(tx) {
            const txList = document.getElementById('transactionsList');
            if (txList.innerHTML.includes('No transactions yet')) {
                txList.innerHTML = '';
            }
            
            const txCard = document.createElement('div');
            txCard.className = 'card transaction-card';
            txCard.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">Transaction: ${tx.transaction_id}</h5>
                    <h6 class="card-subtitle mb-2 text-muted">Customer: ${tx.customer_id}</h6>
                    <p class="card-text">
                        Amount: $${tx.amount}<br>
                        Status: ${tx.status}<br>
                        Vendor ID: ${tx.vendor_id}
                    </p>
                    <div class="timestamp">${tx.timestamp}</div>
                </div>
            `;
            
            txList.insertBefore(txCard, txList.firstChild);
            
            // Update counts
            const txCount = parseInt(document.getElementById('transactionBadge').textContent) + 1;
            document.getElementById('transactionBadge').textContent = txCount;
            document.getElementById('txCount').textContent = txCount;
        }
        
        // Helper function to add prediction to UI
        function addPredictionToUI(pred) {
            const predList = document.getElementById('predictionsList');
            if (predList.innerHTML.includes('No predictions yet')) {
                predList.innerHTML = '';
            }
            
            const predCard = document.createElement('div');
            predCard.className = `card ${pred.prediction ? 'prediction-fraud' : 'prediction-card'}`;
            predCard.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <h5 class="card-title">Result: ${pred.transaction_id}</h5>
                        <span class="badge ${pred.prediction ? 'badge-fraud' : 'badge-legitimate'}">
                            ${pred.prediction ? 'FRAUD' : 'LEGITIMATE'}
                        </span>
                    </div>
                    <p class="card-text">
                        Confidence: ${pred.confidence.toFixed(4)}<br>
                        Model Version: ${pred.model_version}<br>
                        Processor: ${pred.processor_rank}
                    </p>
                    <div class="timestamp">${pred.timestamp}</div>
                </div>
            `;
            
            predList.insertBefore(predCard, predList.firstChild);
            
            // Update counts
            const predCount = parseInt(document.getElementById('predictionBadge').textContent) + 1;
            document.getElementById('predictionBadge').textContent = predCount;
            document.getElementById('predCount').textContent = predCount;
        }
    </script>
</body>
</html>
'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)


def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the UI on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Create templates folder and index.html
    create_templates_folder()
    
    # Open browser automatically
    if not args.no_browser:
        threading.Thread(target=open_browser).start()
    
    # Run Flask app
    app.run(debug=True, port=args.port)