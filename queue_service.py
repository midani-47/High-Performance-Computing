#!/usr/bin/env python
# encoding: utf8

import os
import sys
import json
import time
import signal
import argparse
import traceback
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Constants
QUEUE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a3', 'queue_data')
MOCK_TOKEN = 'mock_token'

# Global flag for graceful shutdown
shutdown_requested = False

# Signal handlers for graceful shutdown
def signal_handler(sig, frame):
    global shutdown_requested
    print(f"Received signal {sig}, shutting down gracefully...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Ensure queue data directory exists
def ensure_queue_data_dir():
    """Ensure the queue data directory exists"""
    try:
        if not os.path.exists(QUEUE_DATA_DIR):
            os.makedirs(QUEUE_DATA_DIR)
            print(f"Created queue data directory: {QUEUE_DATA_DIR}")
        return True
    except Exception as e:
        print(f"Error creating queue data directory: {e}")
        traceback.print_exc()
        return False

# Queue operations
def read_queue(queue_name):
    """Read a queue from disk"""
    try:
        queue_file = os.path.join(QUEUE_DATA_DIR, f"{queue_name}.json")
        if not os.path.exists(queue_file):
            # Create empty queue file if it doesn't exist
            with open(queue_file, 'w') as f:
                json.dump([], f)
            return []
        
        with open(queue_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading queue {queue_name}: {e}")
        traceback.print_exc()
        return []

def write_queue(queue_name, queue_data):
    """Write a queue to disk"""
    try:
        queue_file = os.path.join(QUEUE_DATA_DIR, f"{queue_name}.json")
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f)
        return True
    except Exception as e:
        print(f"Error writing queue {queue_name}: {e}")
        traceback.print_exc()
        return False

def push_message(queue_name, message):
    """Push a message to the queue"""
    try:
        queue = read_queue(queue_name)
        queue.append(message)
        success = write_queue(queue_name, queue)
        return success, len(queue)
    except Exception as e:
        print(f"Error pushing message to queue {queue_name}: {e}")
        traceback.print_exc()
        return False, 0

def pull_message(queue_name):
    """Pull a message from the queue"""
    try:
        queue = read_queue(queue_name)
        if not queue:
            return None
        
        message = queue.pop(0)  # Get the first message (FIFO)
        success = write_queue(queue_name, queue)
        if not success:
            # If writing failed, put the message back
            queue.insert(0, message)
            write_queue(queue_name, queue)
            return None
        
        return message
    except Exception as e:
        print(f"Error pulling message from queue {queue_name}: {e}")
        traceback.print_exc()
        return None

def peek_queue(queue_name):
    """Peek at the first message in the queue without removing it"""
    try:
        queue = read_queue(queue_name)
        if not queue:
            return None
        
        return queue[0]  # Return the first message without removing it
    except Exception as e:
        print(f"Error peeking at queue {queue_name}: {e}")
        traceback.print_exc()
        return None

def clear_queue(queue_name):
    """Clear all messages from the queue"""
    try:
        success = write_queue(queue_name, [])
        return success
    except Exception as e:
        print(f"Error clearing queue {queue_name}: {e}")
        traceback.print_exc()
        return False

def list_queues():
    """List all available queues"""
    try:
        if not os.path.exists(QUEUE_DATA_DIR):
            return []
        
        queues = []
        for filename in os.listdir(QUEUE_DATA_DIR):
            if filename.endswith('.json'):
                queues.append(filename[:-5])  # Remove .json extension
        
        return queues
    except Exception as e:
        print(f"Error listing queues: {e}")
        traceback.print_exc()
        return []

# Authentication middleware
def authenticate():
    """Simple authentication middleware"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False
    
    # In a real implementation, this would validate a proper token
    # For this demo, we just check for a mock token
    token = auth_header.replace('Bearer ', '')
    return token == MOCK_TOKEN

# API Routes
@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "name": "Queue Service API",
        "version": "1.0",
        "endpoints": {
            "/api/queues": "List all queues",
            "/api/queues/<queue_name>/push": "Push a message to a queue",
            "/api/queues/<queue_name>/pull": "Pull a message from a queue",
            "/api/queues/<queue_name>/peek": "Peek at the first message in a queue",
            "/api/queues/<queue_name>/clear": "Clear all messages from a queue"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/queues')
def api_list_queues():
    """List all queues"""
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401
    
    queues = list_queues()
    return jsonify(queues)

@app.route('/api/queues/<queue_name>/push', methods=['POST'])
def api_push_message(queue_name):
    """Push a message to a queue"""
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401
    
    message = request.json
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    success, queue_size = push_message(queue_name, message)
    if success:
        return jsonify({"status": "success", "queue_size": queue_size}), 201
    else:
        return jsonify({"error": "Failed to push message"}), 500

@app.route('/api/queues/<queue_name>/pull', methods=['POST'])
def api_pull_message(queue_name):
    """Pull a message from a queue"""
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401
    
    message = pull_message(queue_name)
    if message is None:
        return jsonify({"error": "Queue is empty"}), 404
    
    return jsonify(message)

@app.route('/api/queues/<queue_name>/peek', methods=['GET'])
def api_peek_queue(queue_name):
    """Peek at the first message in a queue"""
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401
    
    message = peek_queue(queue_name)
    if message is None:
        return jsonify({"error": "Queue is empty"}), 404
    
    return jsonify(message)

@app.route('/api/queues/<queue_name>/clear', methods=['POST'])
def api_clear_queue(queue_name):
    """Clear all messages from a queue"""
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401
    
    success = clear_queue(queue_name)
    if success:
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Failed to clear queue"}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue Service")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the service on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the service to")
    
    args = parser.parse_args()
    
    # Ensure queue data directory exists
    if not ensure_queue_data_dir():
        print("Failed to create queue data directory. Exiting.")
        sys.exit(1)
    
    # Create default queues if they don't exist
    for queue_name in ['TQ1', 'PQ1']:
        queue_file = os.path.join(QUEUE_DATA_DIR, f"{queue_name}.json")
        if not os.path.exists(queue_file):
            with open(queue_file, 'w') as f:
                json.dump([], f)
            print(f"Created queue: {queue_name}")
    
    print(f"Queue Service running on http://{args.host}:{args.port}")
    try:
        # Run Flask app with improved configuration
        app.run(
            host=args.host,
            port=args.port,
            debug=True,  # Keep debug mode for development
            use_reloader=False  # Disable reloader to prevent duplicate processes
        )
    except Exception as e:
        print(f"Error starting Queue Service: {e}")
        traceback.print_exc()
        sys.exit(1)