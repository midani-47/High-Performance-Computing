FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set up working directory
WORKDIR /app

# Install system dependencies including Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-numpy \
    python3-pandas \
    python3-sklearn \
    python3-flask \
    python3-mpi4py \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    python3-joblib \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install any remaining Python packages that don't have system packages
RUN pip install --no-cache-dir python-dotenv requests

# Create necessary directories
RUN mkdir -p /app/mpi
RUN mkdir -p /app/a3/queue_service/queue_data

# Copy application files
COPY prediction_service.py .
COPY .env .

# Copy the model file (assumed to exist at build time)
COPY mpi/fraud_rf_model.pkl ./mpi/

# Start the SSH service for MPI communication
CMD ["python", "prediction_service.py"]
