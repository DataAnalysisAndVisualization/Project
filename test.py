import numpy as np
from index_models import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
import cvxpy as cp
# from data import *
import time



def test_hierarchical_kmeans():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 100)
    vector_ids = list(range(n_vectors))
    root_cluster, _ = calc_hierarchical_kmeans(vectors, vector_ids, 5, 2, max_iter=100)
    for sub_centroids, sub_clusters in zip(root_cluster.sub_centroids, root_cluster.sub_clusters):
        sub_clusters_lens = [sub_cluster.vectors.shape[0] for sub_cluster in sub_clusters.sub_clusters]
        print(sub_centroids.shape, *sub_clusters_lens)

def test_extract_base_clusters():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 100)
    vector_ids = list(range(n_vectors))
    root_cluster, _ = calc_hierarchical_kmeans(vectors, vector_ids, 3, 3, max_iter=100)
    base_centroids, base_clusters = root_cluster.extract_base_clusters()
    print(base_centroids.shape)
    print(len(base_centroids))
    for base_cluster in base_clusters:
        print(base_cluster.cluster_id)

def test_voronoi():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 2)
    ids = list(range(n_vectors))
    root_cluster, _ = calc_hierarchical_kmeans(vectors, ids, 5, 2, max_iter=100)
    lowest_centroids, lowest_clusters = root_cluster.extract_base_clusters()
    vor = Voronoi(lowest_centroids)
    fig = voronoi_plot_2d(vor)
    plt.show()

def plot_voronoi_bad_case():
    vectors = np.array([[0, 0],
                        [0, 1],
                        [0, 4],
                        [1, 3],
                        [1, 4],
                        [0.5, 4.5],
                        [4, 1],
                        [3, -2],
                        [2, -3]])
    vor = Voronoi(vectors)
    fig = voronoi_plot_2d(vor)
    plt.xlim(-5, 5)
    plt.plot((1.3571, 2), (-1.214, 0.5), c='g')
    plt.plot((-0.5, 0.5), (2.5, 4), c='g')
    plt.plot((-0.5, 2), (2.5, 0.5), c='r')
    plt.plot((-0.5, -0.5), (2.5, 0.5), c='b')
    plt.show()

def test_approximate_distance():
    n = 5  # Number of variables in x and y
    m_x = 3  # Number of constraints for x
    m_y = 3  # Number of constraints for y

    # Define variables
    x = cp.Variable(n)
    y = cp.Variable(n)

    # Define problem data (random for illustration)
    A = np.random.randn(m_x, n)
    b = np.random.randn(m_x)

    B = np.random.randn(m_y, n)
    c = np.random.randn(m_y)

    # Objective function: minimize ||x - y||_2^2
    objective = cp.Minimize(cp.sum_squares(x - y))

    # Constraints: Ax <= b and By <= c
    constraints = [A @ x <= b, B @ y <= c]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    print(problem.solve(eps_abs=0.001))
    print("Solver used:", problem.solver_stats.solver_name)
    print("Number of iterations:", problem.solver_stats.num_iters)

def test_calc_polyhedrons():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 2)
    vector_ids = list(range(n_vectors))
    print('kmeans start')
    root_cluster, _ = calc_hierarchical_kmeans(vectors, vector_ids, 5, 3, max_iter=100)
    print('kmeans end')
    lowest_centroids, lowest_clusters = root_cluster.extract_base_clusters()
    As, bs = calc_polyhedrons(lowest_centroids)
    for A, b in zip(As, bs):
        print(A.shape, b.shape)
    print(lowest_centroids.shape, len(As))

    print(As[0])
    print(bs[0])
    t = time.time()
    print(approximate_distance(As[0], bs[0], As[-1], bs[-1], eps_abs=0.001))
    print(time.time()-t)

def test_GreedyKmeans():
    n_vectors = 1000
    dim = 2
    k = 10
    vectors = np.random.rand(n_vectors, dim)
    vector_ids = np.array((range(n_vectors)))
    greedy_kmeans = GreedyKmeans(vectors, vector_ids, n_layer_clusters=20, max_layers=1)
    exhaustive_search = ExhaustiveSearch(vectors, vector_ids)
    x = np.random.rand(dim)

    start = time.time()
    knn_exhaustive = exhaustive_search.knns(x, k)
    print('exhaustive search time: ', time.time() - start)

    start = time.time()
    knn_exhaustive = exhaustive_search.knns(x, k)
    print('exhaustive search time: ', time.time() - start)

    start = time.time()
    knn_exhaustive = exhaustive_search.knns(x, k)
    print('exhaustive search time: ', time.time() - start)

    start = time.time()
    knn = greedy_kmeans.knns(x,k)
    print('GreedyKmeans search time: ', time.time() - start)
    print(x)
    print(knn)

def test_build_mnist_dataset():
    images_np = build_mnist_dataset()
    print(images_np.shape)

def test_build_text_dataset():
    embeddings_array = build_text_dataset()
    print('embedding shape: ', embeddings_array.shape)

def test_compere_models():
    n_vectors = 10000
    dim = 2
    k = 2
    vectors = np.random.rand(n_vectors, dim)
    vector_ids = np.array((range(n_vectors)))
    greedy_kmeans = GreedyKmeans(vectors, vector_ids, n_layer_clusters=60, max_layers=1)
    exhaustive_search = ExhaustiveSearch(vectors, vector_ids)
    models = (greedy_kmeans, exhaustive_search)
    ave_times = compare_models(models, k, vectors, n_compare=10000)
    print(ave_times)


if __name__ == '__main__':
    test_compere_models()


