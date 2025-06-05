•The service will start by reading the pre-trained model fraud_rf_model.pkl that can be found in the examples repository. 
•Then it will read as many requests from the transactions queue (that was implemented in Assignment 3) as processors available. If the queue is empty, the service will block until a message is available. If there are fewer than Prequests in the queue, the service will proceed further with the available requests. 
•For each request, it will pass a message to a worker processor to calculate the result of the final prediction. 
•The service will gather the results of its workers to collect the final predictions, which will be sent to the results queue as individual requests. 
•Now the service will take the next batch of messages from the queue. 
•The number of processors will be configurable and default to 5. We will see MPI examples in the exercises and sample code will be available in the DS_Examples/mpi folder. fraud_rf_model.pkl. 
ALSO check the mpi folder as this is given as reference. 
Take inspiration from the code in the mpi folder and imply for our application as seen in assignment, in the root directory. 
