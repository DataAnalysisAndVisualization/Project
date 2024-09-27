import numpy as np
from index_models import calc_hierarchical_kmeans, extract_lowest_clusters
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def test_hierarchical_kmeans():
    vectors = np.random.rand(1000, 100)
    centroids, clusters = calc_hierarchical_kmeans(vectors, 5, 2, max_iter=100)
    print(centroids.shape)
    for sub_centroids, sub_clusters in clusters:
        sub_clusters_lens = [sub_cluster.shape[0] for sub_cluster in sub_clusters]
        print(sub_centroids.shape, *sub_clusters_lens)

def test_extract_lowest_clusters():
    vectors = np.random.rand(1000, 100)
    centroids, clusters = calc_hierarchical_kmeans(vectors, 5, 3, max_iter=100)
    lowest_centroids, lowest_clusters = extract_lowest_clusters(centroids, clusters)
    print(lowest_centroids.shape)
    print(len(lowest_centroids))

def test_voronoi():
    vectors = np.random.rand(1000, 2)
    centroids, clusters = calc_hierarchical_kmeans(vectors, 5, 2, max_iter=100)
    lowest_centroids, lowest_clusters = extract_lowest_clusters(centroids, clusters)
    vor = Voronoi(lowest_centroids)
    fig = voronoi_plot_2d(vor)
    plt.show()

if __name__ == '__main__':
    test_voronoi()
