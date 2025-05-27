#!/bin/bash
# Bash script for testing the MPI-based fraud detection service

echo -e "\e[32mStarting MPI service test...\e[0m"

# Step 1: Create necessary directories
echo -e "\e[33mCreating necessary directories...\e[0m"
mkdir -p ./mpi
mkdir -p ./a3/queue_service/queue_data

# Step 2: Create model if it doesn't exist
echo -e "\e[33mChecking for model...\e[0m"
if [ ! -f "./mpi/fraud_rf_model.pkl" ]; then
    echo -e "\e[33mModel not found. Creating model...\e[0m"
    python simple_create_model.py
else
    echo -e "\e[32mModel already exists.\e[0m"
fi

# Step 3: Create empty queue files
echo -e "\e[33mCreating empty queue files...\e[0m"
for file in "TQ1.json" "TQ2.json" "PQ1.json"; do
    echo "[]" > "./a3/queue_service/queue_data/$file"
    echo -e "\e[32mCreated empty queue file: ./a3/queue_service/queue_data/$file\e[0m"
done

# Step 4: Create a test transaction
echo -e "\e[33mCreating test transaction...\e[0m"
TEST_TRANSACTION='{
  "transaction_id": "test-tx-001",
  "amount": 5000.00,
  "transaction_count": 25,
  "customer_risk_score": 0.7,
  "vendor_risk_score": 0.6
}'

# Step 5: Add the transaction to the queue
echo -e "\e[33mAdding transaction to queue file...\e[0m"
echo "[$TEST_TRANSACTION]" > ./a3/queue_service/queue_data/TQ1.json

# Step 6: Start the web UI
echo -e "\n\e[36mTo start the web UI, run:\e[0m"
echo -e "\e[47m\e[30mpython web_ui.py\e[0m"

# Step 7: Provide instructions for running the MPI service
echo -e "\n\e[36mTo run the MPI service, execute:\e[0m"
echo -e "\e[47m\e[30mmpirun -n 3 python simple_prediction_service.py\e[0m"

echo -e "\n\e[32mAfter running both commands, visit http://localhost:7600 in your browser\e[0m" 