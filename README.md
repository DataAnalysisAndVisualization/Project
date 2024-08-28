"# Project" 

Idea 1:
Index using multi layer k-means.
Calculate lower bound on distances between groups in k-means.
Use the heuristic for pruning and A* for knn.
Evaluation matrices: index/query time, node expansion count
Datasets: glove, MNIST, NYTimes (https://github.com/erikbern/ann-benchmarks)


Idea 2:
Use the results from RAG in order to rank the documents, and if the results don't match
we will classify the result as hallucinations. we will use information retrivel methods, such as the RM3 model