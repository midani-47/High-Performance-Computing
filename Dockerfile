FROM python:3.9-slim

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    requests \
    python-dotenv

# Copy application files
COPY prediction_service.py .
COPY .env .

# Create mpi directory
RUN mkdir -p mpi

# Copy the model file (assumed to exist at build time)
COPY mpi/fraud_rf_model.pkl ./mpi/

# Start the service
CMD ["python", "prediction_service.py"]
