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
from flask import Flask, render_template, request, jsonify, redirect, url_for

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
    "num_processors": int(os.getenv("NUM_PROCESSORS", "3"))
}

# Log configuration for debugging
logger.info(f"Queue Data Directory: {CONFIG['queue_data_dir']}")
logger.info(f"Transaction Queue Files: {CONFIG['transaction_queue_file']}, {CONFIG['transaction_queue_file2']}")
logger.info(f"Results Queue File: {CONFIG['results_queue_file']}")

app = Flask(__name__)

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
        
        logger.info(f"SimpleQueueClient initialized with data directory: {self.queue_data_dir}")
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
    
    def get_queue_size(self, queue_file):
        """Get the number of messages in a queue."""
        try:
            messages = self._read_queue(queue_file)
            return len(messages)
        except Exception as e:
            logger.error(f"Error getting queue size for {queue_file}: {str(e)}")
            return 0
    
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
    
    def get_all_messages(self, queue_file):
        """Get all messages from a queue file without removing them."""
        return self._read_queue(queue_file)

def generate_sample_transaction():
    """Generate a sample transaction for testing."""
    transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
    amount = random.uniform(100, 10000)
    transaction_count = random.randint(1, 50)
    customer_risk_score = random.uniform(0, 1)
    vendor_risk_score = random.uniform(0, 1)
    
    # Create a transaction with the required fields for the fraud detection model
    return {
        "transaction_id": transaction_id,
        "amount": round(amount, 2),
        "transaction_count": transaction_count,
        "customer_risk_score": round(customer_risk_score, 2),
        "vendor_risk_score": round(vendor_risk_score, 2)
    }

# Initialize the queue client
queue_client = SimpleQueueClient(CONFIG)

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
                          results_queue_size=results_queue_size,
                          transaction_queue=CONFIG["transaction_queue_file"],
                          transaction_queue2=CONFIG["transaction_queue_file2"],
                          results_queue=CONFIG["results_queue_file"])

@app.route('/push_transactions', methods=['POST'])
def push_transactions():
    """Push sample transactions to the transaction queue."""
    try:
        # Get form data
        count = int(request.form.get('num_transactions', '1'))
        if count < 1 or count > 100:
            count = 5  # Default to 5 if invalid
        
        queue_choice = request.form.get('target_queue', 'TQ1')
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
            return jsonify({"success": False, "message": "Failed to push transactions"})
    except Exception as e:
        logger.error(f"Error in push_transactions: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/get_results')
def get_results():
    """Get prediction results from the results queue."""
    try:
        results = queue_client.get_all_messages(queue_client.results_queue_file)
        return jsonify({
            "success": True,
            "count": len(results),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/queue_status')
def queue_status():
    """Get the current status of all queues."""
    try:
        # Get queue sizes
        transaction_queue_size = queue_client.get_queue_size(queue_client.transaction_queue_file)
        transaction_queue2_size = queue_client.get_queue_size(queue_client.transaction_queue_file2)
        results_queue_size = queue_client.get_queue_size(queue_client.results_queue_file)
        
        return jsonify({
            "success": True,
            "transaction_queue_size": transaction_queue_size,
            "transaction_queue2_size": transaction_queue2_size,
            "results_queue_size": results_queue_size
        })
    except Exception as e:
        logger.error(f"Error in queue_status: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/force_refresh')
def force_refresh():
    """Force refresh the page."""
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(CONFIG["queue_data_dir"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=7600, debug=True) 