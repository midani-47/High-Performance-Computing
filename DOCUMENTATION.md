# Technical Documentation: MPI-Based Fraud Detection Service

## Authors
- Abed Midani
- Nevin Joseph

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [MPI Implementation](#mpi-implementation)
4. [Queue Integration](#queue-integration)
5. [Prediction Model](#prediction-model)
6. [Data Flow](#data-flow)
7. [Error Handling](#error-handling)
8. [Containerization](#containerization)
9. [Performance Considerations](#performance-considerations)
10. [Limitations and Future Improvements](#limitations-and-future-improvements)

## Introduction

This document provides technical details about an MPI-based machine learning prediction service designed to process transaction data for fraud detection. The service utilizes the Message Passing Interface (MPI) to distribute prediction tasks across multiple processors or nodes, enabling high-performance parallel processing.

The prediction service was developed as part of Assignment 4, building upon the message queue service from Assignment 3. It serves as a computation-intensive component in a distributed system architecture, demonstrating how MPI can be used to scale machine learning workloads.

## System Architecture

The prediction service follows a master-worker architecture pattern:

1. **Master Process**: Coordinates the distribution of work, communicates with the queue service, and collects results from workers.
2. **Worker Processes**: Receive transaction data, make predictions using the pre-trained model, and return results to the master.

### Technology Stack

The system is built using the following technologies:

- **MPI4Py**: Python bindings for the Message Passing Interface (MPI) standard, enabling parallel computation across multiple processors.
- **OpenMPI**: An open-source implementation of the MPI standard used as the underlying MPI framework.
- **Docker**: The application is containerized using Docker, making it easy to deploy. Docker Compose is used to orchestrate multiple containers for a distributed deployment.
- **Scikit-learn**: Used for loading and utilizing the pre-trained Random Forest model for fraud detection.
- **Pandas/NumPy**: Used for data manipulation and preprocessing of transaction data.
- **Requests**: For communication with the queue service API.
- **Python-dotenv**: For configuration management through environment variables.

## MPI Implementation

### Process Hierarchy

The MPI implementation follows a master-worker pattern:

- **Rank 0 Process (Master)**: Responsible for:
  - Authenticating with the queue service
  - Loading the ML model
  - Pulling transactions from the transaction queue
  - Distributing work to worker processes
  - Collecting prediction results
  - Pushing results to the prediction queue

- **Rank 1+ Processes (Workers)**: Responsible for:
  - Loading the ML model locally
  - Receiving transaction data from the master
  - Preprocessing transaction data
  - Making predictions using the model
  - Returning prediction results to the master

### Communication Pattern

The communication between processes follows a synchronous pattern:

1. Master sends transaction data to workers in a round-robin fashion
2. Workers process the data and send back results
3. Master collects all results before proceeding to the next batch

This approach ensures that all transactions in a batch are processed before moving to the next batch, maintaining the order of transactions as much as possible.

### Code Structure

The main components of the MPI implementation are:

```python
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Master process logic
if rank == 0:
    # Pull transactions from queue
    # Distribute to workers
    # Collect results
    # Push results to queue
else:
    # Worker process logic
    # Receive transaction
    # Process and make prediction
    # Send result back to master
```

## Queue Integration

The prediction service integrates with the message queue service from Assignment 3 through a RESTful API client. The `QueueClient` class handles:

- **Authentication**: Using JWT tokens for secure access
- **Message Pulling**: Retrieving transaction messages from the transaction queue
- **Message Pushing**: Sending prediction results to the prediction queue
- **Error Handling**: Managing HTTP errors, token expiration, and retries

The integration follows these principles:

1. **Decoupling**: The queue service and prediction service are completely decoupled, communicating only through the queue interface
2. **Asynchronous Processing**: Transactions can be added to the queue at any time and will be processed as resources become available
3. **Persistence**: The queue service handles persistence, ensuring no transactions are lost

## Prediction Model

The service uses a pre-trained Random Forest model for fraud detection:

- **Model Loading**: The model is loaded from a pickle file (`fraud_rf_model.pkl`)
- **Input Processing**: Transaction data is preprocessed to extract relevant features
- **Prediction**: The model outputs a binary classification (fraud/not fraud) and a confidence score
- **Result Formatting**: Predictions are formatted with transaction ID, prediction result, confidence score, and metadata

## Data Flow

The complete data flow through the system is as follows:

1. **Initialization**:
   - Master and worker processes initialize
   - Each process loads the ML model
   - Master authenticates with the queue service

2. **Transaction Processing**:
   - Master pulls up to N transactions from the queue (N = number of workers)
   - Master distributes transactions to workers
   - Workers preprocess transactions and make predictions
   - Workers send prediction results back to master
   - Master pushes results to the prediction queue

3. **Cycle Continuation**:
   - Master proceeds to the next batch of transactions
   - If the queue is empty, master waits and retries

## Error Handling

The service implements robust error handling:

- **Model Loading Errors**: If the model fails to load, the process logs the error and exits
- **Queue Communication Errors**: If communication with the queue fails, the service logs the error and retries
- **Authentication Errors**: If authentication fails, the service logs the error and retries or exits if persistent
- **Prediction Errors**: If prediction fails for a transaction, the error is logged and the transaction is skipped
- **MPI Communication Errors**: If MPI communication fails, the service logs the error and attempts to recover or gracefully exit

## Containerization

The service is containerized using Docker to ensure consistent deployment across environments:

- **Base Image**: Ubuntu 22.04 with Python 3 and OpenMPI
- **SSH Configuration**: Set up for communication between MPI nodes
- **Volume Mounting**: The model file is mounted to avoid rebuilding the image when the model changes
- **Network Configuration**: A custom bridge network enables communication between containers

Docker Compose is used to orchestrate multiple containers:

- **Master Node**: Coordinates the MPI processes and communicates with the queue service
- **Worker Nodes**: Process transactions in parallel
- **Network**: All nodes are connected through a dedicated Docker network

## Performance Considerations

Several design decisions were made to optimize performance:

- **Parallel Processing**: Distributing work across multiple processors reduces overall processing time
- **Batched Processing**: Processing transactions in batches reduces communication overhead
- **Local Model Loading**: Each process loads the model locally to avoid model serialization/deserialization overhead
- **Efficient Communication**: Only transaction data and prediction results are communicated between processes
- **Asynchronous Queue Integration**: The service doesn't block on queue operations, allowing for continuous processing

## Limitations and Future Improvements

Current limitations and potential improvements include:

- **Dynamic Scaling**: Currently, the number of processes is fixed at startup. Future implementations could dynamically adjust based on load.
- **Load Balancing**: The round-robin distribution doesn't account for varying transaction complexity. A more sophisticated load balancing strategy could be implemented.
- **Fault Tolerance**: If a worker fails, its assigned transactions won't be processed. A checkpoint/restart mechanism could be added.
- **Model Updating**: The model is loaded once at startup. A mechanism for hot-swapping models could be implemented.
- **Performance Monitoring**: Adding detailed performance metrics and monitoring would help identify bottlenecks.
- **Security Enhancements**: While basic authentication is implemented, additional security measures like TLS for API communication would improve security.
