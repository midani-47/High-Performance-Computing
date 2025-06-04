#!/usr/bin/env python
# encoding: utf8

import json
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import argparse

# Constants
MODEL_FILENAME = 'mpi/fraud_rf_model.pkl'
STATUSES = ['submitted', 'accepted', 'rejected']
TEST_DATA_FILE = 'mpi/synthetic_test_data.csv'


def load_model(filename=MODEL_FILENAME):
    """Load the pre-trained fraud detection model"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    return joblib.load(filename)


def load_test_data(filename=TEST_DATA_FILE, num_samples=10):
    """Load test data from CSV file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Test data file {filename} not found.")
    
    df = pd.read_csv(filename)
    
    # Select a random subset of the data
    if num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
    
    # Convert DataFrame to list of dictionaries
    transactions = df.to_dict('records')
    
    # Add transaction_id field
    for i, tx in enumerate(transactions):
        tx['transaction_id'] = f"test-tx-{i+1}"
    
    return transactions


def preprocess_transaction(transaction):
    """Preprocess a transaction for prediction"""
    # Create a DataFrame with a single row
    df = pd.DataFrame([transaction])
    
    # Convert timestamp to Unix timestamp if it's a string
    if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
    
    # Encode status using the same order as training
    if 'status' in df.columns:
        status_mapping = {status: idx for idx, status in enumerate(STATUSES)}
        df['status'] = df['status'].map(status_mapping)
    
    # Drop columns not used for prediction
    columns_to_drop = ['fraudulent', 'customer_id', 'transaction_id'] 
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df


def predict_fraud(model, transaction):
    """Predict if a transaction is fraudulent"""
    # Preprocess the transaction
    X = preprocess_transaction(transaction)
    
    # Make prediction
    prediction = bool(model.predict(X)[0])
    confidence = float(np.max(model.predict_proba(X)[0]))
    
    # Create result object
    result = {
        "transaction_id": transaction.get("transaction_id", "unknown"),
        "prediction": prediction,
        "confidence": confidence,
        "model_version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "processor_rank": 0  # Single process test
    }
    
    return result


def main(num_samples=10, verbose=False):
    """Test the fraud detection model with sample data"""
    print(f"Testing fraud detection with {num_samples} samples")
    
    # Load the model
    model = load_model()
    print("Model loaded successfully")
    
    # Load test data
    transactions = load_test_data(num_samples=num_samples)
    print(f"Loaded {len(transactions)} test transactions")
    
    # Process each transaction
    results = []
    for i, transaction in enumerate(transactions):
        if verbose:
            print(f"\nProcessing transaction {i+1}/{len(transactions)}:")
            print(json.dumps(transaction, indent=2))
        
        result = predict_fraud(model, transaction)
        results.append(result)
        
        if verbose:
            print("Prediction result:")
            print(json.dumps(result, indent=2))
        else:
            status = "FRAUD" if result["prediction"] else "LEGITIMATE"
            print(f"Transaction {result['transaction_id']}: {status} (confidence: {result['confidence']:.4f})")
    
    # Print summary
    fraud_count = sum(1 for r in results if r["prediction"])
    print(f"\nSummary: {fraud_count} fraudulent transactions detected out of {len(results)}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Fraud Detection Model")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples to process")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output for each transaction")
    
    args = parser.parse_args()
    
    main(num_samples=args.samples, verbose=args.verbose)