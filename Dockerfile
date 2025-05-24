FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages - split into multiple commands to avoid memory issues
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir pandas scikit-learn
RUN pip install --no-cache-dir joblib flask
RUN pip install --no-cache-dir python-dotenv requests
RUN pip install --no-cache-dir mpi4py

# Set environment variables for MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Create necessary directories
RUN mkdir -p /app/mpi
RUN mkdir -p /app/a3/queue_service/queue_data

# Copy application files
COPY prediction_service.py /app/
COPY create_model.py /app/
COPY .env /app/

# Initialize empty queue files
RUN echo '[]' > /app/a3/queue_service/queue_data/TQ1.json
RUN echo '[]' > /app/a3/queue_service/queue_data/TQ2.json
RUN echo '[]' > /app/a3/queue_service/queue_data/PQ1.json
RUN chmod -R 777 /app/a3/queue_service/queue_data

# Set environment variables for MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Entry point to create model and start the service
CMD ["bash", "-c", "python create_model.py && mpirun --allow-run-as-root -n 6 python prediction_service.py"]
