import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data import build_mnist_dataset, build_text_dataset, random_data
from index_models import ExhaustiveSearch, GreedyKmeans, OurKDtree, OurBallTree, calc_polyhedrons, calc_distance_matrix


def time_model_search_vector(model, vector, k, n_tries):
    sum_time = 0
    for i in range(n_tries):
        start_time = time.time()
        model.knns(vector, k)
        end_time = time.time()
        sum_time += end_time - start_time
    return sum_time/n_tries

def plot_search_time_boxplot(models, vectors, k, n_tries, dataset_name):
    data = []
    for model in models:
        for vector in vectors:
            search_time = time_model_search_vector(model, vector, k, n_tries)
            data.append({'model': model.name, 'search_time': search_time})

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='search_time', data=df)
    plt.title(f'{k}-nns over {dataset_name}')
    plt.xlabel('model')
    plt.ylabel('Search Time (sec)')
    plt.savefig(f'{dataset_name}_{k}_nns_search_time.png')
    plt.show()



def plot_dataset(dataset, dataset_name):
    np.random.seed(42)
    vector_ids = np.array(range(dataset.shape[0]))
    exhaustive_search = ExhaustiveSearch(dataset, vector_ids)
    greedy_kmeans = GreedyKmeans(dataset, vector_ids, n_layer_clusters=40, max_layers=1)
    kd_tree = OurKDtree(dataset, vector_ids)
    ball_tree = OurBallTree(dataset, vector_ids)
    models = [exhaustive_search, greedy_kmeans, kd_tree, ball_tree]
    search_vectors = dataset[np.random.randint(dataset.shape[0], size=500)]
    plot_search_time_boxplot(models, search_vectors, 10, 2,
                             dataset_name)

def plot_searched():
    np.random.seed(42)
    dims = [2,3,5,10]
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    for dim in dims:
        uniform_set = random_data(10000, dim)
        vector_ids = np.array(range(uniform_set.shape[0]))
        greedy_kmeans = GreedyKmeans(uniform_set, vector_ids, n_layer_clusters=40, max_layers=1)
        search_vectors = uniform_set[np.random.randint(uniform_set.shape[0], size=500)]
        ave_search_list = []
        for k in ks:
            sum_searched = 0
            for vector in search_vectors:
                _, _, searched = greedy_kmeans.knns_with_count(vector, k)
                sum_searched += searched
            ave_searched = sum_searched/len(search_vectors)
            ave_search_list.append(ave_searched)
        plt.plot(ks, ave_search_list, label=dim)
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('Average number of searched clusters')
    plt.legend(title="data dimension")
    plt.title('Average number of searched clusters vs k')
    plt.savefig('searched_clusters.png')
    plt.show()


def cluster_searched():
    # check the time for finding the avarage distance between the clusters compared to the number of clusters
    np.random.seed(42)
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    dims = [2,3,5,10]
    for dim in dims:
        time_list = []
        t = time.time()
        uniform_set = random_data(10000, dim)
        vector_ids = np.array(range(uniform_set.shape[0]))
        greedy_kmeans = GreedyKmeans(uniform_set, vector_ids, n_layer_clusters=40, max_layers=1)
        ave_distance_list = []
        for k in ks:
            sum_distance = 0
            for vector in uniform_set:
                _, distances, _ = greedy_kmeans.knns_with_count(vector, k)
                sum_distance += np.mean(distances)
            ave_distance = sum_distance/len(uniform_set)
            ave_distance_list.append(ave_distance)
            time_list.append(time.time()-t)
            t = time.time()
        plt.plot(ks, time_list, label=dim)
    # plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('Time')
    plt.legend(title="dim")
    plt.title('Time vs k')
    plt.savefig('time_vs_k.png')
    plt.show()

def plot_ave_dist_time():
    n_redu = 10
    np.random.seed(42)
    uniform_set = random_data(10000, 3)
    vector_ids = np.array(range(uniform_set.shape[0]))
    ks = list(range(2, 31))
    ave_times = []
    for k in ks:
        greedy_kmeans = GreedyKmeans(uniform_set, vector_ids, n_layer_clusters=k, max_layers=1)
        As, bs = calc_polyhedrons(greedy_kmeans.base_centroids)
        start_time = time.time()
        for i in range(n_redu):
            calc_distance_matrix(As, bs, greedy_kmeans.base_centroids, 0.001)
        end_time = time.time()
        n = len(greedy_kmeans.base_centroids)
        ave_time = (end_time - start_time) / (n*(n-1)*n_redu/2)
        ave_times.append(ave_time)

    plt.plot(ks, ave_times)
    plt.xlabel('Clusters number')
    plt.ylabel('Time (sec)')
    plt.title('Average distance approximation time vs clusters number')
    plt.savefig('ave_dist_time_vs_cluster_num')
    plt.show()


def main():
    np.random.seed(42)
    # mnist = build_mnist_dataset()[:2000]
    # plot_dataset(mnist, 'mnist')
    # wiki = build_text_dataset()
    # plot_dataset(wiki, 'miniLM embedded wiki')
    # uniform_set = random_data(10000, 2)
    # plot_dataset(uniform_set, 'uniform dataset')
    # cluster_searched()
    plot_ave_dist_time()

if __name__ == '__main__':
    main()
