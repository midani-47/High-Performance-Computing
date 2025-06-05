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
        queue_service_status['last_checked'] = datetime.now(timezone.utc).isoformat()
        queue_service_status['url'] = QUEUE_SERVICE_URL
        print(f"Queue service health check: {queue_service_status['healthy']} at {QUEUE_SERVICE_URL}")
        return queue_service_status['healthy']
    except requests.exceptions.RequestException as e:
        print(f"Queue service connection error: {e}")
        queue_service_status['healthy'] = False
        queue_service_status['last_checked'] = datetime.now(timezone.utc).isoformat()
        queue_service_status['url'] = QUEUE_SERVICE_URL
        queue_service_status['error'] = str(e)
        return False
    except Exception as e:
        print(f"Unexpected error checking queue service health: {e}")
        queue_service_status['healthy'] = False
        queue_service_status['last_checked'] = datetime.now(timezone.utc).isoformat()
        queue_service_status['url'] = QUEUE_SERVICE_URL
        queue_service_status['error'] = str(e)
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
                print("Max retries reached. Queue service unavailable.")
                return []
        except Exception as e:
            print(f"Unexpected error fetching predictions: {e}")
            traceback.print_exc()
            return []
    
    return results


@app.route('/')
def index():
    """Render the main page"""
    # Check queue service health
    check_queue_service_health()
    return render_template('index.html', 
                           transactions=transactions, 
                           predictions=predictions,
                           queue_status=queue_service_status,
                           queue_service_url=QUEUE_SERVICE_URL)


@app.route('/submit', methods=['POST'])
def submit_transaction():
    """Submit a transaction"""
    try:
        # Get form data - handle both form data and JSON data
        if request.is_json:
            data = request.get_json()
            customer_id = data.get('customer_id', f"cust-{random.randint(1000, 9999)}")
            amount = float(data.get('amount', 0))
            vendor_id = int(data.get('vendor_id', 0))
            status = data.get('status', 'submitted')
        else:
            customer_id = request.form.get('customer_id', f"cust-{random.randint(1000, 9999)}")
            amount = float(request.form.get('amount', 0))
            vendor_id = int(request.form.get('vendor_id', 0))
            status = request.form.get('status', 'submitted')
        
        # Validate inputs
        if amount <= 0:
            return jsonify({"success": False, "error": "Amount must be greater than 0"}), 400
        if vendor_id <= 0:
            return jsonify({"success": False, "error": "Vendor ID must be greater than 0"}), 400
        
        # Create transaction
        transaction = {
            "transaction_id": f"tx-{int(time.time())}-{random.randint(1000, 9999)}",
            "customer_id": customer_id,
            "amount": amount,
            "vendor_id": vendor_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
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
    except ValueError as e:
        print(f"Validation error submitting transaction: {e}")
        return jsonify({"success": False, "error": f"Invalid input: {str(e)}"}), 400
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
                "timestamp": datetime.now(timezone.utc).isoformat()
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
    """Health check endpoint that also checks queue service health"""
    # Check queue service health
    queue_healthy = check_queue_service_health()
    
    return jsonify({
        "status": "healthy",
        "queue_service": {
            "healthy": queue_healthy,
            "url": QUEUE_SERVICE_URL,
            "last_checked": queue_service_status.get('last_checked'),
            "error": queue_service_status.get('error')
        }
    }), 200





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