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
    "num_processors": int(os.getenv("NUM_PROCESSORS", "5"))
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
    
    def clear_queue(self, queue_file):
        """Clear all messages from a queue file."""
        try:
            with open(queue_file, 'w') as f:
                json.dump([], f)
            logger.info(f"Cleared queue file: {queue_file}")
            return True
        except Exception as e:
            logger.error(f"Error clearing queue file {queue_file}: {str(e)}")
            return False

def generate_sample_transaction():
    """Generate a sample transaction for testing.
    
    Creates a transaction in the Assignment 3 message format:
    {
      "content": {
        "transaction_id": str,
        "customer_id": str,
        "customer_name": str,
        "amount": float,
        "vendor_id": str,
        "date": str,
        ... other transaction fields
      },
      "timestamp": datetime,
      "message_type": "transaction",
      "id": str
    }
    """
    transaction_id = f"tx_{uuid.uuid4().hex[:8]}"
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    
    # Generate a random amount with some variance
    amount = round(random.uniform(100, 10000), 2)
    
    # Create transaction content
    transaction_content = {
        "transaction_id": transaction_id,
        "customer_id": f"cust_{uuid.uuid4().hex[:6]}",
        "customer_name": f"Customer {random.randint(1, 100)}",
        "amount": amount,
        "vendor_id": f"vendor_{uuid.uuid4().hex[:6]}",
        "date": datetime.now().isoformat(),
        "transaction_count": random.randint(1, 50),
        "customer_risk_score": round(random.uniform(0, 1), 2),
        "vendor_risk_score": round(random.uniform(0, 1), 2)
    }
    
    # Create the full message in Assignment 3 format
    return {
        "content": transaction_content,
        "timestamp": datetime.now().isoformat(),
        "message_type": "transaction",
        "id": message_id
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
    
    # Get some sample transactions and results for preview
    sample_transactions = []
    if transaction_queue_exists and transaction_queue_size > 0:
        all_transactions = queue_client.get_all_messages(queue_client.transaction_queue_file)
        sample_transactions = all_transactions[:5]  # Show up to 5 transactions
    
    sample_results = []
    if results_queue_exists and results_queue_size > 0:
        all_results = queue_client.get_all_messages(queue_client.results_queue_file)
        sample_results = all_results[:5]  # Show up to 5 results
    
    return render_template('index.html', 
                          transaction_queue_exists=transaction_queue_exists,
                          transaction_queue_size=transaction_queue_size,
                          transaction_queue2_exists=transaction_queue2_exists,
                          transaction_queue2_size=transaction_queue2_size,
                          results_queue_exists=results_queue_exists,
                          results_queue_size=results_queue_size,
                          transaction_queue=CONFIG["transaction_queue_file"],
                          transaction_queue2=CONFIG["transaction_queue_file2"],
                          results_queue=CONFIG["results_queue_file"],
                          sample_transactions=sample_transactions,
                          sample_results=sample_results,
                          num_processors=CONFIG["num_processors"])

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
        
        # Get some sample transactions and results for preview
        sample_transactions = []
        if transaction_queue_size > 0:
            all_transactions = queue_client.get_all_messages(queue_client.transaction_queue_file)
            sample_transactions = all_transactions[:5]  # Show up to 5 transactions
        
        sample_results = []
        if results_queue_size > 0:
            all_results = queue_client.get_all_messages(queue_client.results_queue_file)
            sample_results = all_results[:5]  # Show up to 5 results
        
        return jsonify({
            "success": True,
            "transaction_queue_size": transaction_queue_size,
            "transaction_queue2_size": transaction_queue2_size,
            "results_queue_size": results_queue_size,
            "sample_transactions": sample_transactions,
            "sample_results": sample_results
        })
    except Exception as e:
        logger.error(f"Error in queue_status: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/clear_queue', methods=['POST'])
def clear_queue():
    """Clear a specific queue."""
    try:
        queue_name = request.form.get('queue_name', '')
        
        if queue_name == 'TQ1':
            queue_file = queue_client.transaction_queue_file
        elif queue_name == 'TQ2':
            queue_file = queue_client.transaction_queue_file2
        elif queue_name == 'PQ1':
            queue_file = queue_client.results_queue_file
        else:
            return jsonify({"success": False, "message": "Invalid queue selection"})
        
        success = queue_client.clear_queue(queue_file)
        
        if success:
            return jsonify({
                "success": True, 
                "message": f"Successfully cleared queue {queue_name}"
            })
        else:
            return jsonify({"success": False, "message": f"Failed to clear queue {queue_name}"})
    except Exception as e:
        logger.error(f"Error in clear_queue: {str(e)}")
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