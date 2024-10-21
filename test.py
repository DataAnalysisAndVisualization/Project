import numpy as np
from index_models import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
import cvxpy as cp
import time


def test_hierarchical_kmeans():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 100)
    ids = list(range(n_vectors))
    centroids, clusters = calc_hierarchical_kmeans(vectors, ids, 5, 2, max_iter=100)
    print(centroids.shape)
    for sub_centroids, sub_clusters in clusters:
        sub_clusters_lens = [sub_cluster.shape[0] for sub_cluster in sub_clusters]
        print(sub_centroids.shape, *sub_clusters_lens)

def test_extract_lowest_clusters():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 100)
    ids = list(range(n_vectors))
    centroids, clusters = calc_hierarchical_kmeans(vectors, ids, 5, 3, max_iter=100)
    lowest_centroids, lowest_clusters = extract_lowest_clusters(centroids, clusters)
    print(lowest_centroids.shape)
    print(len(lowest_centroids))

def test_voronoi():
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, 2)
    ids = list(range(n_vectors))
    centroids, clusters = calc_hierarchical_kmeans(vectors, ids, 5, 2, max_iter=100)
    lowest_centroids, lowest_clusters = extract_lowest_clusters(centroids, clusters)
    vor = Voronoi(lowest_centroids)
    fig = voronoi_plot_2d(vor)
    plt.show()
    # print(vor.ridge_points)
    # print(vor.ridge_vertices)

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

# Support function: Find the point on the convex hull farthest in a given direction
# Equals to the farthest vertex in the case of a polygon
def support(points, direction):
    best_point = points[0]
    best_distance = np.dot(best_point, direction)

    for point in points[1:]:
        distance = np.dot(point, direction)
        if distance > best_distance:
            best_distance = distance
            best_point = point

    return best_point


# Minkowski difference support function for two convex hulls
def support_minkowski(points1, points2, direction):
    p1 = support(points1, direction)
    p2 = support(points2, -direction)
    return p1 - p2


# Project vector 'v' onto vector 'u'
def project(v, u):
    return (np.dot(v, u) / np.dot(u, u)) * u


# Function to check if the origin is inside the simplex and update the direction
def contains_origin(simplex, direction):
    A = simplex[-1]
    AO = -A

    # Compute projections for each face of the simplex
    for i in range(len(simplex) - 1):
        B = simplex[i]
        AB = B - A
        projection_AB_AO = project(AO, AB)
        if np.linalg.norm(projection_AB_AO) < np.linalg.norm(AO):  # AO is not fully aligned with AB
            direction[:] = AO - projection_AB_AO  # Move perpendicular to AB
            simplex[:] = [A, B]
            return False
    return True

# Main GJK algorithm for arbitrary dimensions
def gjk(points1, points2):
    # Start with an arbitrary direction, e.g., (1, 0, ..., 0) for n dimensions
    dim = points1.shape[1]
    direction = np.ones(dim)

    # Initial point in the Minkowski difference
    simplex = [support_minkowski(points1, points2, direction)]

    # Reverse direction towards the origin
    direction = -simplex[0]

    while True:
        # Add a new point in the Minkowski difference along the current direction
        new_point = support_minkowski(points1, points2, direction)

        # If the new point is not farther than the origin, return the distance
        if np.dot(new_point, direction) <= 0:
            return False, np.linalg.norm(direction)  # Distance found

        # Add the new point to the simplex
        simplex.append(new_point)

        # Check if the origin is within the new simplex
        if contains_origin(simplex, direction):
            return True, 0  # The origin is within the simplex, meaning there's an intersection

def test_gjk():
    # Example in 4D space
    points1 = np.array([[1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]])  # Convex hull 1 (in 4D)
    points2 = np.array([[2, 2, 2, 2], [2, 0, 2, 2], [0, 2, 2, 2], [0, 0, 2, 2]])  # Convex hull 2 (in 4D)

    # Call GJK to find distance in 4D space
    intersection, distance = gjk(points1, points2)
    if intersection:
        print("Convex hulls intersect")
    else:
        print(f"Distance between convex hulls: {distance}")


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
    ids = list(range(n_vectors))
    print('kmeans start')
    centroids, clusters = calc_hierarchical_kmeans(vectors, ids, 5, 3, max_iter=100)
    print('kmeans end')
    lowest_centroids, lowest_clusters = extract_lowest_clusters(centroids, clusters)
    As, bs = calc_polyhedrons(lowest_centroids)
    for A, b in zip(As, bs):
        print(A.shape, b.shape)
    print(lowest_centroids.shape, len(As))

    print(As[0])
    print(bs[0])
    t = time.time()
    print(approximate_distance(As[0], bs[0], As[-1], bs[-1], eps_abs=0.001))
    print(time.time()-t)

if __name__ == '__main__':
    test_calc_polyhedrons()


