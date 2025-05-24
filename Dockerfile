FROM python:3.9-slim

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openssh-server \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for MPI
RUN mkdir -p /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#StrictHostKeyChecking ask/StrictHostKeyChecking no/' /etc/ssh/ssh_config

# Generate SSH keys
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -N ""
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# Install Python dependencies with specific versions to ensure compatibility
RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    requests==2.31.0 \
    python-dotenv==1.1.0 \
    mpi4py==4.0.3 \
    joblib==1.5.1 \
    flask==3.1.1

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
