import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data import build_mnist_dataset
from index_models import ExhaustiveSearch


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

def plot_search_time_boxplot(model, vectors, ks, n_tries):
    data = []
    for k in ks:
        for vector in vectors:
            search_time = time_model_search_vector(model, vector, k, n_tries)
            data.append({'k': k, 'search_time': search_time})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Plot the boxplot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='k', y='search_time', data=df)
    plt.title('Search Time by k')
    plt.xlabel('k')
    plt.ylabel('Search Time')
    plt.show()


def main():
    np.random.seed(42)
    mnist_dataset = build_mnist_dataset()
    vector_ids = np.array(range(mnist_dataset.shape[0]))
    exhaustive_search = ExhaustiveSearch(mnist_dataset, vector_ids)
    search_vectors = mnist_dataset[np.random.randint(mnist_dataset.shape[0],size=100)]
    plot_search_time_boxplot(exhaustive_search, search_vectors, ks=[1,2,3], n_tries=3)

if __name__ == '__main__':
    main()
