import numpy as np
from sklearn.metrics import pairwise_distances

"""
Receive the vectors as a nparray.
calculate hierarchical kmeans:
do kmeans and redo for every cluster. Calc outside of the class.
Extract lowest level clusters.
Create proximity graph from them.
Calculate Voronoi diagram on lowest level.
Calculate cluster lower bound using voronoi diagram.

Search alg:
find closest lowest level cluster.
Execute greedy nns search.


"""
class GreedyKmeans:
    def __init__(self, vectors, dim, n_layer_clusters, max_layers):
        pass


def chose_kmeans_pp_clusters(vectors, n_clusters):
    """
    Initialize k-means-plus-plus clusters
    """
    # Choose the first centroid uniformly
    # Choose vectors to be centroids,
    # with bigger probability the farther they are from previously chosen centroids
    n_clusters = min(vectors.shape[0], n_clusters)
    centroids = np.zeros((n_clusters, vectors.shape[1]))
    centroids[0] = vectors[np.random.choice(vectors.shape[0])]  # Choose the first centroid randomly
    for i in range(1, n_clusters):
        distances = np.zeros(vectors.shape[0])
        for j, vector in enumerate(vectors):
            distances[j] = np.min(np.linalg.norm(vector - centroids[:i], axis=1))
        probabilities = distances**2 / np.sum(distances**2)
        next_centroid_idx = np.random.choice(np.arange(len(vectors)), p=probabilities)
        centroids[i] = vectors[next_centroid_idx]
    return centroids


def label_vectors_to_centroids(vectors, centroids):
    """
    Given vectors and centroids, separate the vectors to an index based on the centroids
    :return: A list of 2d-arrays
    """
    distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    clusters = {}
    for vector, label in zip(vectors, labels):
        if label in clusters:
            clusters[label].append(vector)
        else:
            clusters[label] = [vector, ]

    for label in clusters:
        clusters[label] = np.array(clusters[label])

    return [clusters[label] for label in range(centroids.shape[0])]


def update_k_means_centroids(vectors, centroids):
    """
    Update the k-means centroids
    """
    n_clusters = centroids.shape[0]
    distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    new_centroids = []
    for cluster_id in range(n_clusters):
        cluster_vectors = vectors[labels == cluster_id]
        # remove empty clusters
        if len(cluster_vectors) > 0:
            new_centroid = cluster_vectors.mean(axis=0)
            new_centroids.append(new_centroid)

    return np.array(new_centroids)

def calc_k_means(vectors, n_clusters, max_iter=100):
    """
    Calculate k-means-plus-plus clusters
    :param vectors:
    :param n_clusters: The number of clusters
    :param max_iter: Max iterations number
    :return: The centroids, the vectors divided into clusters
    """
    iteration = 0
    centroids = chose_kmeans_pp_clusters(vectors, n_clusters)
    prev_centroids = centroids
    centroids = update_k_means_centroids(vectors, centroids)
    while iteration < max_iter and not np.array_equal(centroids, prev_centroids):
        prev_centroids = centroids
        centroids = update_k_means_centroids(vectors, centroids)
        iteration += 1

    clusters = label_vectors_to_centroids(vectors, centroids)

    return centroids, clusters

def calc_hierarchical_kmeans(vectors, n_clusters, max_layers, max_iter=100):
    """
    :param vectors:
    :param n_clusters:
    :param max_layers:
    :param max_iter:
    :return: a list of lists of length 2 and depth n_clusters.
    The first value is the centroids and the second value is the sub-clusters recursively.
    [top_centroids, [...[sub_centroids_1, sub_cluster_1], ..., [sub_centroids_n, sub_cluster_n]...]
    """
    centroids, clusters = calc_k_means(vectors, n_clusters, max_iter)
    if max_layers > 1:
        sub_clusters = [calc_hierarchical_kmeans(clusters[cluster_idx], n_clusters, max_layers-1, max_iter)
                       for cluster_idx in range(n_clusters)]
        return centroids, sub_clusters

    return centroids, clusters

def extract_lowest_clusters(centroids, clusters):
    if type(clusters[0]) == np.ndarray:
        return centroids, clusters

    lowest_centroids = None
    lowest_clusters = []
    for cluster in clusters:
        sub_centroids, sub_clusters = cluster
        cluster_lowest_centroids, cluster_lowest_clusters = extract_lowest_clusters(sub_centroids, sub_clusters)
        if lowest_centroids is not None:
            lowest_centroids = np.concatenate((lowest_centroids, cluster_lowest_centroids), axis=0)
        else:
            lowest_centroids = cluster_lowest_centroids
        lowest_clusters.extend(cluster_lowest_clusters)

    return lowest_centroids, lowest_clusters