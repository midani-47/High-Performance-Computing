FROM ubuntu:22.04

# Install system dependencies including OpenMPI
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY prediction_service.py .
COPY .env .

# Copy the model file (assumed to exist at build time)
COPY mpi/fraud_rf_model.pkl ./mpi/

# Set up SSH for MPI
RUN mkdir -p /var/run/sshd
RUN echo 'StrictHostKeyChecking no' >> /etc/ssh/ssh_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

# Add the SSH key
RUN mkdir -p /root/.ssh
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -N ""
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# Expose SSH port
EXPOSE 22

# Start the service
CMD ["python3", "prediction_service.py"]
