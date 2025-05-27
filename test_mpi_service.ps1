# PowerShell script for testing the MPI-based fraud detection service
Write-Host "Starting MPI service test..." -ForegroundColor Green

# Step 1: Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
if (-not (Test-Path "./mpi")) {
    New-Item -Path "./mpi" -ItemType Directory | Out-Null
}
if (-not (Test-Path "./a3/queue_service/queue_data")) {
    New-Item -Path "./a3/queue_service/queue_data" -ItemType Directory -Force | Out-Null
}

# Step 2: Create model if it doesn't exist
Write-Host "Checking for model..." -ForegroundColor Yellow
if (-not (Test-Path "./mpi/fraud_rf_model.pkl")) {
    Write-Host "Model not found. Creating model..." -ForegroundColor Yellow
    python simple_create_model.py
} else {
    Write-Host "Model already exists." -ForegroundColor Green
}

# Step 3: Create empty queue files
Write-Host "Creating empty queue files..." -ForegroundColor Yellow
@("TQ1.json", "TQ2.json", "PQ1.json") | ForEach-Object {
    $filePath = "./a3/queue_service/queue_data/$_"
    Set-Content -Path $filePath -Value "[]"
    Write-Host "Created empty queue file: $filePath" -ForegroundColor Green
}

# Step 4: Create a test transaction
Write-Host "Creating test transaction..." -ForegroundColor Yellow
$testTransaction = @{
    transaction_id = "test-tx-001"
    amount = 5000.00
    transaction_count = 25
    customer_risk_score = 0.7
    vendor_risk_score = 0.6
} | ConvertTo-Json -Compress

# Step 5: Add the transaction to the queue
Write-Host "Adding transaction to queue file..." -ForegroundColor Yellow
Set-Content -Path "./a3/queue_service/queue_data/TQ1.json" -Value "[$testTransaction]"

# Step 6: Start the web UI
Write-Host "`nTo start the web UI, run:" -ForegroundColor Cyan
Write-Host "python web_ui.py" -ForegroundColor White -BackgroundColor DarkBlue

# Step 7: Provide instructions for running the MPI service
Write-Host "`nTo run the MPI service, execute:" -ForegroundColor Cyan
Write-Host "mpiexec -n 3 python simple_prediction_service.py" -ForegroundColor White -BackgroundColor DarkBlue

Write-Host "`nAfter running both commands, visit http://localhost:7600 in your browser" -ForegroundColor Green 