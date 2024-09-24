import numpy as np
from index_models import calc_hierarchical_kmeans

if __name__ == '__main__':
    vectors = np.random.rand(1000, 100)
    centroids, clusters = calc_hierarchical_kmeans(vectors, 5, 2, max_iter=100)
    print(centroids.shape)
    for sub_centroids, sub_clusters in clusters:
        sub_clusters_lens = [sub_cluster.shape[0] for sub_cluster in sub_clusters]
        print(sub_centroids.shape, *sub_clusters_lens)