import time
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import numpy as np
import cvxpy as cp
import timeit

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


what to do?
calc hierarchical kmeans with object.
at lowest level store centroid, vectors, vector ids, centroid id.

"""

class BaseCluster:
    def __init__(self, vectors, vector_ids, centroid, cluster_id):
        self.vectors = vectors
        self.vector_ids = vector_ids
        self.centroid = centroid
        self.cluster_id = cluster_id
        self.layer=0

    def update_top_k(self, x, k, top_k_vector_ids, top_k_vector_distances):
        """
        Update the top k closest neighbors to x by scanning the cluster
        """
        cluster_distances = np.linalg.norm(self.vectors - x, axis=1)# cdist(x, self.vectors, 'euclidean')
        concat_distances = np.concatenate((top_k_vector_distances, cluster_distances))
        concat_ids = np.concatenate((top_k_vector_ids, self.vector_ids))
        if k <= top_k_vector_ids.shape[0] + self.vectors.shape[0]:
            new_top_k_idx = np.argpartition(concat_distances, k)[:k]
            return concat_ids[new_top_k_idx], concat_distances[new_top_k_idx]
        else:
            return concat_ids, concat_distances

    def find_top_k(self, x, k):
        cluster_distances = np.linalg.norm(self.vectors - x, axis=1)# cdist(x, self.vectors, 'euclidean')
        if k < self.vectors.shape[0]:
            top_k_idx = np.argpartition(cluster_distances, k)[:k]
            return self.vector_ids[top_k_idx], cluster_distances[top_k_idx]

        return self.vector_ids, cluster_distances

class SuperCluster:
    def __init__(self, centroid, sub_centroids, sub_clusters, layer):
        self.centroid = centroid
        self.sub_centroids = sub_centroids
        self.sub_clusters = sub_clusters
        self.layer = layer

    def extract_base_clusters(self):
        if self.layer == 1:
            return self.sub_centroids, self.sub_clusters

        base_centroids = None
        base_clusters = []
        for cluster in self.sub_clusters:
            cluster_base_centroids, cluster_base_clusters = cluster.extract_base_clusters()
            if base_centroids is not None:
                base_centroids = np.concatenate((base_centroids, cluster_base_centroids), axis=0)
            else:
                base_centroids = cluster_base_centroids
            base_clusters.extend(cluster_base_clusters)

        return base_centroids, base_clusters

    def find_closest_cluster(self, x):
        x_cluster_distances = np.linalg.norm(self.sub_centroids - x, axis=1)# cdist(x.reshape(1, -1), self.sub_centroids, 'euclidean')
        closest_index = np.argmin(x_cluster_distances)
        return self.sub_clusters[closest_index]


def calc_hierarchical_kmeans(vectors, vector_ids, n_clusters, max_layers, max_iter=100, cluster_id=0, super_centroid=None):
    centroids, clusters = calc_k_means(vectors, vector_ids, n_clusters, max_iter)
    if max_layers > 1:
        sub_clusters = []
        for i in range(centroids.shape[0]):
            sub_cluster_vectors, sub_cluster_vector_ids = clusters[i]
            sub_centroid = centroids[i]
            sub_cluster, cluster_id = calc_hierarchical_kmeans(sub_cluster_vectors, sub_cluster_vector_ids, n_clusters,
                                                               max_layers-1, max_iter=100, cluster_id=cluster_id,
                                                               super_centroid=sub_centroid)
            sub_clusters.append(sub_cluster)
        super_cluster = SuperCluster(super_centroid, centroids, sub_clusters, layer=max_layers)
        return super_cluster, cluster_id

    base_clusters = []
    for i in range(centroids.shape[0]):
        base_centroid = centroids[i]
        base_vectors, base_vector_ids = clusters[i]
        base_cluster = BaseCluster(base_vectors, base_vector_ids, base_centroid, cluster_id)
        base_clusters.append(base_cluster)
        cluster_id += 1

    super_cluster = SuperCluster(super_centroid, centroids, base_clusters, layer=max_layers)
    return super_cluster, cluster_id

class GreedyKmeans:
    def __init__(self, vectors, vector_ids, n_layer_clusters, max_layers):
        self.root_cluster, _ = calc_hierarchical_kmeans(vectors, vector_ids, n_layer_clusters, max_layers, max_iter=100)
        self.base_centroids, self.base_clusters = self.root_cluster.extract_base_clusters()
        As, bs = calc_polyhedrons(self.base_centroids)
        self.queues = calc_search_queues(As, bs, self.base_centroids, self.base_clusters, eps_abs=0.001)

    def find_closest_base_cluster(self, x):
        cluster = self.root_cluster
        while cluster.layer > 0:
            cluster = cluster.find_closest_cluster(x)
        return cluster

    def knns(self, x, k):
        # delete_this = 1
        closest_cluster = self.find_closest_base_cluster(x)
        top_k_vector_ids, top_k_vector_distances = closest_cluster.find_top_k(x, k)
        cluster_queue = self.queues[closest_cluster.cluster_id]
        for clusters_distance, cluster in cluster_queue:
            if clusters_distance > max(top_k_vector_distances):
                # print('searched: ', delete_this, '/', len(cluster_queue)+1)
                return top_k_vector_ids, top_k_vector_distances
            # delete_this += 1
            top_k_vector_ids, top_k_vector_distances = cluster.update_top_k(x, k, top_k_vector_ids, top_k_vector_distances)
        # print('searched: ', delete_this, '/', len(cluster_queue)+1)
        return top_k_vector_ids, top_k_vector_distances

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

def label_vectors_to_centroids(vectors, ids, centroids):
    """
    Given vectors and centroids, separate the vectors to an index based on the centroids
    :return: A list of 2d-arrays
    """
    distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    clusters = {}
    for vector, id_, label in zip(vectors, ids, labels):
        if label in clusters:
            clusters[label][0].append(vector)
            clusters[label][1].append(id_)
        else:
            clusters[label] = [[vector, ], [id_,]]

    for label in clusters:
        clusters[label][0] = np.array(clusters[label][0])
        clusters[label][1] = np.array(clusters[label][1])

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

def calc_k_means(vectors, ids, n_clusters, max_iter=100):
    """
    Calculate k-means-plus-plus clusters
    :param vectors:
    :param ids:
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

    clusters = label_vectors_to_centroids(vectors, ids, centroids)

    return centroids, clusters

def calc_polyhedrons(points):
    """
    Calculate the half-spaces that define the voronoi polyhedrons
    :return: list[A], list[b]
    """
    npoints = points.shape[0]
    dir_vecs = [[] for _ in range(npoints)]
    project_vals = [[] for _ in range(npoints)]

    for point1id, point1 in enumerate(points):
        for point2id, point2 in enumerate(points):
            if point1id <= point2id:
                continue
            mean_point = np.mean((point1,point2), axis=0)
            dir_vec = point2 - point1
            project_val = np.dot(dir_vec, mean_point)
            dir_vecs[point1id].append(dir_vec)
            project_vals[point1id].append(project_val)

            dir_vecs[point2id].append(-1*dir_vec)
            project_vals[point2id].append(-1*project_val)

    As = [np.array(dir_vecs_) for dir_vecs_ in dir_vecs]
    bs = [np.array(project_vals_) for project_vals_ in project_vals]
    for i, (A, b) in enumerate(zip(As, bs)):
        # A, b = simplify_polyhedron(A,b)
        As[i], bs[i] = A, b
    return As, bs

def approximate_distance(A, b, x0, B, c, y0, eps_abs=0.001):
    """
    Approximate the distance between 2 polyhedrons.
    min ||x-y||_2^2 s.t. Ax<=b and By<=c
    """
    n = A.shape[1]
    # Define variables
    x = cp.Variable(n)
    y = cp.Variable(n)

    x.value = x0
    y.value = y0

    # Objective function: minimize ||x - y||_2^2
    objective = cp.Minimize(cp.sum_squares(x - y))

    # Constraints: Ax <= b and By <= c
    constraints = [A @ x <= b, B @ y <= c]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    # try:
    solution = problem.solve(solver='SCS', eps_abs=eps_abs)# , solver='OSQP' TODO: Understand why OSQP doesn't work for dim=2
    # except Exception as e:
    #     print(A@x0<=b)
    #     print(B@y0<=c)
    #     print("x is affine:", x.is_affine())
    #     print("y is affine:", y.is_affine())
    #     print("Problem is DCP:", problem.is_dcp())
    #     print("Problem is DPP:", problem.is_dpp())
    #
    #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    #     raise e
    return solution-eps_abs
    # return problem.solve(eps_abs=eps_abs)

def calc_search_queues(As, bs, base_centroids, base_clusters, eps_abs=0.001):
    distance_matrix = calc_distance_matrix(As, bs, base_centroids, eps_abs)
    n_clusters = len(base_clusters)
    queues = []
    for i in range(n_clusters):
        queue = []
        for j in range(n_clusters):
            if i == j:
                continue
            cluster = base_clusters[j]
            dist = distance_matrix[i, j]
            queue.append((dist, cluster))
        queue.sort(key=lambda x: x[0])
        queues.append(queue)
    return queues

def calc_distance_matrix(As, bs, centroids, eps_abs=0.001):
    n_polyhedrons = len(As)
    distances = np.zeros((n_polyhedrons, n_polyhedrons))
    for i in range(n_polyhedrons):
        A, b, x0 = As[i], bs[i], centroids[i]
        for j in range(i):
            B, c, y0 = As[j], bs[j], centroids[j]
            dist = approximate_distance(A, b, x0, B, c, y0, eps_abs)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

class ExhaustiveSearch:
    def __init__(self, vectors, vector_ids):
        self.name = 'exhaustive search'
        self.vectors = vectors
        self.vector_ids = vector_ids

    def knns(self, x, k):
        distances = np.linalg.norm(self.vectors - x, axis=1)  # cdist(x, self.vectors, 'euclidean')
        top_k_idx = np.argpartition(distances, k)[:k]
        return self.vector_ids[top_k_idx], distances[top_k_idx]

def compare_models(models,k, vectors, n_compare):
    sum_time = np.zeros(len(models))
    for i in range(n_compare):
        x = vectors[np.random.randint(vectors.shape[0])]
        for j, model in enumerate(models):
            time_start = time.time()
            model.knns(x,k)
            time_end = time.time()
            sum_time[j] += time_end-time_start
    return sum_time / n_compare

class KDtree:
    def __init__(self, vectors, vector_ids):
        self.name = 'KD-tree'
        self.kdtree = KDTree(vectors)
        self.vectors = vectors
        self.vector_ids = vector_ids

    def knns(self, x, k):
        distances, top_k_idx = self.kdtree.query(x, k)
        return self.vector_ids[top_k_idx], distances
    
class BallTree:
    def __init__(self, vectors, vector_ids):
        self.name = 'Ball-tree'
        self.balltree = BallTree(vectors)
        self.vectors = vectors
        self.vector_ids = vector_ids

    def knns(self, x, k):
        distances, top_k_idx = self.balltree.query(x, k)
        return self.vector_ids[top_k_idx], distances
