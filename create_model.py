#!/usr/bin/env python3
"""
Script to create a simple random forest model for fraud detection.
This is just for demonstration purposes.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Create directory for model if it doesn't exist
os.makedirs('mpi', exist_ok=True)

# Generate synthetic data for fraud detection
np.random.seed(42)
n_samples = 1000

# Features: amount, transaction_count, time_since_last_transaction, vendor_risk_score
X = np.random.rand(n_samples, 4)
X[:, 0] = X[:, 0] * 10000  # amount between 0 and 10000
X[:, 1] = np.random.randint(1, 100, size=n_samples)  # transaction_count between 1 and 100
X[:, 2] = np.random.randint(1, 1000, size=n_samples)  # time_since_last_transaction in minutes
X[:, 3] = np.random.rand(n_samples)  # vendor_risk_score between 0 and 1

# Target: 0 for legitimate, 1 for fraudulent (5% fraud rate)
y = np.zeros(n_samples)
fraud_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
y[fraud_indices] = 1

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=['amount', 'transaction_count', 'time_since_last_transaction', 'vendor_risk_score'])
df['is_fraud'] = y

print("Generated synthetic data:")
print(df.head())
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Model accuracy on training data: {train_score:.4f}")
print(f"Model accuracy on test data: {test_score:.4f}")

# Save the model
model_path = 'mpi/fraud_rf_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Create a simple function to demonstrate how to use the model
def predict_fraud(transaction):
    """Predict if a transaction is fraudulent."""
    # Extract features from transaction
    features = np.array([
        transaction.get('amount', 0),
        transaction.get('transaction_count', 1),
        transaction.get('time_since_last_transaction', 0),
        transaction.get('vendor_risk_score', 0.5)
    ]).reshape(1, -1)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'transaction_id': transaction.get('transaction_id', 'unknown')
    }

# Test the function with a sample transaction
sample_transaction = {
    'transaction_id': 'tx123',
    'amount': 5000,
    'transaction_count': 10,
    'time_since_last_transaction': 120,
    'vendor_risk_score': 0.3
}

print("\nTesting prediction function:")
prediction_result = predict_fraud(sample_transaction)
print(f"Transaction ID: {prediction_result['transaction_id']}")
print(f"Is fraud: {prediction_result['is_fraud']}")
print(f"Fraud probability: {prediction_result['fraud_probability']:.4f}")
