import numpy as np
from sklearn.metrics import pairwise_distances
import sympy as sp
import cvxpy as cp


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
    def __init__(self, vectors, ids, dim, n_layer_clusters, max_layers):
        self.centroids, self.clusters = calc_hierarchical_kmeans(vectors, ids, n_layer_clusters, max_layers, max_iter=100)
        self.lowest_centroids, self.lowest_clusters = extract_lowest_clusters(self.centroids, self.clusters)
        As, bs = calc_polyhedrons(self.lowest_centroids)
        queues = calc_search_queues(As, bs, self.lowest_clusters, eps_abs=0.001)

    def find_closest_cluster(self, x):
        pass


    def knns(self, x, k):
        closest_cluster, closest_cluster_id = self.find_closest_cluster(x)
        top_k = update_top_k(x, k, [], closest_cluster)
        queue = queues[closest_cluster_id]
        # for dist,



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

def calc_hierarchical_kmeans(vectors, ids, n_clusters, max_layers, max_iter=100):
    """
    :param vectors:
    :param ids:
    :param n_clusters:
    :param max_layers:
    :param max_iter:
    :return: a list of lists of length 2 and depth n_clusters.
    The first value is the centroids and the second value is the sub-clusters recursively.
    [top_centroids, [...[sub_centroids_1, sub_cluster_1], ..., [sub_centroids_n, sub_cluster_n]...]
    [bottom_centroids, [[vectors_1, ids_1], ..., [vectors_n, ids_n]]
    """
    centroids, clusters = calc_k_means(vectors, ids, n_clusters, max_iter)
    if max_layers > 1:
        sub_clusters = [calc_hierarchical_kmeans(clusters[cluster_idx][0], clusters[cluster_idx][1], n_clusters, max_layers-1, max_iter)
                       for cluster_idx in range(n_clusters)]
        return centroids, sub_clusters

    return centroids, clusters

def extract_lowest_clusters(centroids, clusters):
    if type(clusters[0][1][0]) == int:# if lowest level (sub_cluster_1[0] vs ids_1[0])
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

def calc_voronoi_polyhedrons(vor):
    """
    Calculate the half-spaces that define the voronoi polyhedrons
    :param vor:
    :return: list[A], list[b]
    """
    npoints = vor.points.shape[0]
    dir_vecs = []
    project_vals = []
    for i in range(npoints):
        dir_vecs.append([])
        project_vals.append([])

    for point1id, point2id in vor.ridge_points:
        point1, point2 = vor.points[[point1id, point2id]]
        mean_point = np.mean((point1,point2), axis=0)
        dir_vec = point2 - point1
        project_val = np.dot(dir_vec, mean_point)
        dir_vecs[point1id].append(dir_vec)
        project_vals[point1id].append(project_val)

        dir_vecs[point2id].append(-1*dir_vec)
        project_vals[point2id].append(-1*project_val)

    As = [np.array(dir_vecs_) for dir_vecs_ in dir_vecs]
    bs = [np.array(project_vals_) for project_vals_ in project_vals]
    return As, bs

def simplify_polyhedron(A,b):
    M = np.column_stack((A, b))
    M_sympy = sp.Matrix(M)
    ref, _ = M_sympy.rref()
    ref = np.array(ref.tolist()).astype(np.float64)
    n_non_zero_rows = np.max(np.nonzero(ref)[0])
    ref = ref[:n_non_zero_rows+1]
    A_new = ref[:,:-1]
    b_new = ref[:,-1]
    return A_new, b_new


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


def approximate_distance(A, b, B, c, eps_abs=0.001):
    """
    Approximate the distance between 2 polyhedrons.
    min ||x-y||_2^2 s.t. Ax<=b and By<=c
    """
    n = A.shape[1]
    # Define variables
    x = cp.Variable(n)
    y = cp.Variable(n)

    # Objective function: minimize ||x - y||_2^2
    objective = cp.Minimize(cp.sum_squares(x - y))

    # Constraints: Ax <= b and By <= c
    constraints = [A @ x <= b, B @ y <= c]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    return problem.solve(eps_abs=eps_abs, solver='OSQP')

def calc_search_queues(As, bs, lowest_clusters, eps_abs=0.001):
    distance_matrix = calc_distance_matrix(As, bs, eps_abs)
    n_clusters = len(lowest_clusters)
    queues = []
    for i in range(n_clusters):
        queue = []
        for j in range(n_clusters):
            if i == j:
                continue
            cluster = lowest_clusters[j]
            dist = distance_matrix[i, j]
            queue.append((dist, cluster))
        queue.sort()
        queues.append(queue)
    return queues

def calc_distance_matrix(As, bs, eps_abs=0.001):
    n_polyhedrons = len(As)
    distances = np.zeros((n_polyhedrons, n_polyhedrons))
    for i in range(n_polyhedrons):
        A, b = As[i], bs[i]
        for j in range(i):
            B, c = As[j], bs[j]
            dist = approximate_distance(A, b, B, c, eps_abs)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances


def update_top_k(x, k, top_k, cluster):
    """
    Update the top k closest neighbors to x by scanning the cluster
    """
    cluster_vectors, cluster_ids = cluster
    cluster_distances = cp.spatial.distance.cdist(x, cluster_vectors, 'euclidean').tolist()
    return (top_k + list(zip(cluster_distances, cluster_ids))).sort()[:k]