# Technical Documentation: MPI-Based Fraud Detection Service

## Authors
- Nevin Joseph
- Abed Midani

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [MPI Implementation](#mpi-implementation)
4. [Queue Integration](#queue-integration)
5. [Prediction Model](#prediction-model)
6. [Data Flow](#data-flow)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Limitations and Future Improvements](#limitations-and-future-improvements)

## Introduction

This document provides technical details about an MPI-based machine learning prediction service designed to process transaction data for fraud detection. The service utilizes the Message Passing Interface (MPI) to distribute prediction tasks across multiple processors, enabling high-performance parallel processing.

The prediction service was developed as part of Assignment 4, building upon the queue files from Assignment 3. It serves as a computation-intensive component in a distributed system architecture, demonstrating how MPI can be used to scale machine learning workloads.

## System Architecture

The prediction service follows a master-worker architecture pattern:

1. **Master Process**: Coordinates the distribution of work, reads from queue files, and collects results from workers.
2. **Worker Processes**: Receive transaction data, make predictions using the pre-trained model, and return results to the master.

### Technology Stack

The system is built using the following technologies:

- **MPI4Py**: Python bindings for the Message Passing Interface (MPI) standard, enabling parallel computation across multiple processors.
- **OpenMPI**: An open-source implementation of the MPI standard used as the underlying MPI framework.
- **Scikit-learn**: Used for loading and utilizing the pre-trained Random Forest model for fraud detection.
- **Pandas/NumPy**: Used for data manipulation and preprocessing of transaction data.

## MPI Implementation

### Process Hierarchy

The MPI implementation follows a master-worker pattern:

- **Rank 0 Process (Master)**: Responsible for:
  - Reading from the transaction queue files
  - Distributing work to worker processes
  - Collecting prediction results
  - Writing results to the prediction queue file

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
    # Pull transactions from queue files
    # Distribute to workers
    # Collect results
    # Push results to results queue file
else:
    # Worker process logic
    # Receive transaction
    # Process and make prediction
    # Send result back to master
```

## Queue Integration

The prediction service integrates with the queue files from Assignment 3:

- **File-Based Queues**: The system uses JSON files in the `a3/queue_service/queue_data` directory to store transactions and prediction results.
- **Transaction Queues**: TQ1.json and TQ2.json store incoming transactions.
- **Results Queue**: PQ1.json stores prediction results.

The `SimpleQueueClient` class handles:
- Reading transactions from queue files
- Writing prediction results to the results file
- Managing queue file access and modifications

## Prediction Model

The service uses a pre-trained Random Forest model for fraud detection:

- **Model Loading**: The model is loaded from a pickle file (`mpi/fraud_rf_model.pkl`)
- **Input Processing**: Transaction data is preprocessed to extract relevant features
- **Prediction**: The model outputs a binary classification (fraud/not fraud) and a confidence score
- **Result Formatting**: Predictions are formatted with transaction ID, prediction result, confidence score, and metadata

## Data Flow

The complete data flow through the system is as follows:

1. **Initialization**:
   - Master and worker processes initialize
   - Each process loads the ML model

2. **Transaction Processing**:
   - Master reads up to N transactions from the queue files (N = number of workers)
   - Master distributes transactions to workers
   - Workers preprocess transactions and make predictions
   - Workers send prediction results back to master
   - Master writes results to the prediction results file

3. **Cycle Continuation**:
   - Master proceeds to the next batch of transactions
   - If the queues are empty, master waits and retries

## Error Handling

The service implements robust error handling:

- **Model Loading Errors**: If the model fails to load, a mock model is used as fallback
- **Queue File Errors**: If reading/writing to queue files fails, the service logs the error and retries
- **Prediction Errors**: If prediction fails for a transaction, the error is logged and the transaction is skipped
- **MPI Communication Errors**: If MPI communication fails, the service logs the error and attempts to recover

## Performance Considerations

Several design decisions were made to optimize performance:

- **Parallel Processing**: Distributing work across multiple processors reduces overall processing time
- **Batched Processing**: Processing transactions in batches reduces communication overhead
- **Local Model Loading**: Each process loads the model locally to avoid model serialization/deserialization overhead
- **Efficient Communication**: Only transaction data and prediction results are communicated between processes
- **Queue Prioritization**: The service checks both transaction queues and processes from the one with more messages first

## Limitations and Future Improvements

Current limitations and potential improvements include:

- **Dynamic Scaling**: Currently, the number of processes is fixed at startup. Future implementations could dynamically adjust based on load.
- **Load Balancing**: The round-robin distribution doesn't account for varying transaction complexity. A more sophisticated load balancing strategy could be implemented.
- **Fault Tolerance**: If a worker fails, its assigned transactions won't be processed. A checkpoint/restart mechanism could be added.
- **Model Updating**: The model is loaded once at startup. A mechanism for hot-swapping models could be implemented.
- **Performance Monitoring**: Adding detailed performance metrics and monitoring would help identify bottlenecks.
