import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data import build_mnist_dataset
from index_models import ExhaustiveSearch, GreedyKmeans


def time_model_search_vector(model, vector, k, n_tries):
    sum_time = 0
    for i in range(n_tries):
        start_time = time.time()
        model.knns(vector, k)
        end_time = time.time()
        sum_time += end_time - start_time
    return sum_time/n_tries

def time_model_search_vectors(model, vectors, k, n_tries):
    times = [time_model_search_vector(model, vector, k, n_tries) for vector in vectors]
    return times

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
    plt.ylabel('Search Time')
    plt.show()
    plt.savefig(f'{dataset_name}_{k}_nns_search_time.png')



def main():
    np.random.seed(42)
    mnist_dataset = build_mnist_dataset()[:10000]
    vector_ids = np.array(range(mnist_dataset.shape[0]))
    exhaustive_search = ExhaustiveSearch(mnist_dataset, vector_ids)
    greedy_kmeans = GreedyKmeans(mnist_dataset, vector_ids, n_layer_clusters=20, max_layers=1)
    search_vectors = mnist_dataset[np.random.randint(mnist_dataset.shape[0],size=500)]
    plot_search_time_boxplot([exhaustive_search, greedy_kmeans], search_vectors, 10, 2,
                             'mnist')

if __name__ == '__main__':
    main()
