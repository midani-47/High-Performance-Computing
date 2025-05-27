#!/usr/bin/env python3
"""
Simple Model Generator

This script creates a Random Forest model for fraud detection and saves it to a pickle file.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_synthetic_data(n_samples=1000):
    """Create synthetic fraud detection data."""
    np.random.seed(42)
    
    # Generate features
    amount = np.random.normal(5000, 2000, n_samples)
    transaction_count = np.random.randint(1, 100, n_samples)
    customer_risk_score = np.random.random(n_samples)
    vendor_risk_score = np.random.random(n_samples)
    
    # Generate target (5% fraud rate)
    is_fraud = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    is_fraud[fraud_indices] = 1.0
    
    # Create DataFrame
    data = pd.DataFrame({
        'amount': amount,
        'transaction_count': transaction_count,
        'customer_risk_score': customer_risk_score,
        'vendor_risk_score': vendor_risk_score,
        'is_fraud': is_fraud
    })
    
    print("Generated synthetic data:")
    print(data.head())
    print(f"Fraud rate: {data['is_fraud'].mean():.2%}")
    
    return data

def train_model(data):
    """Train a Random Forest model on the data."""
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Model accuracy on training data: {train_acc:.4f}")
    print(f"Model accuracy on test data: {test_acc:.4f}")
    
    return model

def save_model(model, model_path):
    """Save the model to a pickle file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")

def test_model(model_path):
    """Test loading and using the model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model successfully loaded for verification")
        
        # Create a test transaction
        test_transaction = {
            'transaction_id': 'tx123',
            'amount': 1000.0,
            'transaction_count': 5,
            'customer_risk_score': 0.3,
            'vendor_risk_score': 0.2
        }
        
        # Extract features
        features = np.array([
            test_transaction['amount'],
            test_transaction['transaction_count'],
            test_transaction['customer_risk_score'],
            test_transaction['vendor_risk_score']
        ]).reshape(1, -1)
        
        # Make prediction
        probs = model.predict_proba(features)[0]
        is_fraud = bool(probs[1] > 0.5)
        
        print("\nTesting prediction function:")
        print(f"Transaction ID: {test_transaction['transaction_id']}")
        print(f"Is fraud: {is_fraud}")
        print(f"Fraud probability: {probs[1]:.4f}")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")

def main():
    """Main function to create and save the model."""
    model_path = "./mpi/fraud_rf_model.pkl"
    
    # Skip if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        test_model(model_path)
        return
    
    # Create synthetic data
    data = create_synthetic_data()
    
    # Train model
    model = train_model(data)
    
    # Save model
    save_model(model, model_path)
    
    # Test model
    test_model(model_path)

if __name__ == "__main__":
    main() 