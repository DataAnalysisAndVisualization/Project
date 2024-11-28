import torch
from torchvision import datasets, transforms
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer



def build_mnist_dataset(download=True):
    # Define the transformations to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Download the MNIST dataset
    mnist_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=download)
    # Convert all images to a numpy array
    images_np = np.array([np.array(mnist_dataset[i][0]).squeeze().flatten() for i in range(len(mnist_dataset))])
    # print("mnist dataset from shape: "+images_np.shape)
    return images_np



def build_text_dataset(DATASET_NAME='wikitext', CONFIG_NAME='wikitext-2-raw-v1', EMBEDDING_MODEL='all-MiniLM-L6-v2',
                       n_rows=None):
    # Load the pre-trained embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    dataset = load_dataset(DATASET_NAME, name=CONFIG_NAME)
    texts = dataset['train']['text']
    if n_rows is not None:
        texts = texts[:n_rows]
    # Generate embeddings for the texts
    embeddings = model.encode(texts)
    embeddings_array = np.array(embeddings)
    # texts_array = np.array(texts)

    # Check the shape of the embeddings and a few text samples
    # print("Embeddings shape:", embeddings_array.shape)
    # print(embeddings_array[0], texts_array[0])
    return embeddings_array # Return the NumPy arrays

def random_data(n_vectors, dim):
    # generate a random dataset that contains points between 0 and 1
    data = np.random.rand(n_vectors, dim)
    return data

