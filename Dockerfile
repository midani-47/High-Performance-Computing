FROM python:3.9-slim

# Set up working directory
WORKDIR /app

# Install system dependencies and Python packages in a single layer to reduce build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    openmpi-bin \
    libopenmpi-dev \
    python3-mpi4py \
    && pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    flask \
    python-dotenv \
    requests \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && echo 'root:password' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#StrictHostKeyChecking ask/StrictHostKeyChecking no/' /etc/ssh/ssh_config \
    && mkdir -p /root/.ssh \
    && ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" \
    && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys

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
