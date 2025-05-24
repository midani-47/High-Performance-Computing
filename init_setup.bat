@echo off
echo Setting up environment for MPI Prediction Service...

REM Create necessary directories
mkdir a3\queue_service\queue_data 2>nul
mkdir mpi 2>nul

REM Generate model if it doesn't exist
if not exist mpi\fraud_rf_model.pkl (
    echo Generating fraud detection model...
    python create_model.py
)

REM Create empty queue files if they don't exist
if not exist a3\queue_service\queue_data\TQ1.json (
    echo Creating empty TQ1.json...
    echo [] > a3\queue_service\queue_data\TQ1.json
)

if not exist a3\queue_service\queue_data\TQ2.json (
    echo Creating empty TQ2.json...
    echo [] > a3\queue_service\queue_data\TQ2.json
)

if not exist a3\queue_service\queue_data\PQ1.json (
    echo Creating empty PQ1.json...
    echo [] > a3\queue_service\queue_data\PQ1.json
)

echo Environment setup complete!
echo You can now run: docker-compose up --build -d
