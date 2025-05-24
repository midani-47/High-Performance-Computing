FROM continuumio/miniconda3:latest

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

# Create a conda environment with all required packages
RUN conda create -n mpi_env python=3.9 -y && \
    conda install -n mpi_env -c conda-forge mpi4py numpy pandas scikit-learn joblib flask python-dotenv requests -y

# Make RUN commands use the conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "source activate mpi_env" > ~/.bashrc
ENV PATH /opt/conda/envs/mpi_env/bin:$PATH

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
