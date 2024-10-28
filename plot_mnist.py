from parametric_umap import ParametricUMAP
from utils import MLP
import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

# Convert the dataset to a numpy array of shape (60000, 784)
data = trainset.data.numpy().reshape(-1, 784)
# Normalize the data to have zero mean and unit variance
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-7)

# encoder = MLP(784, [5000, 5000, 5000, 5000, 512], 2).to(torch.device('cuda'))
# model = ParametricUMAP(784, 2, data, encoder, K=15).to(torch.device('cuda'))
# model.load_state_dict(torch.load('./mnist_model.pth', weights_only=True))

# # Plot the embedded MNIST dataset
# import matplotlib.pyplot as plt
# embedding = model.transform(torch.tensor(data, dtype=torch.float).to(model.device)).detach().cpu().numpy()
# plt.scatter(embedding[:, 0], embedding[:, 1], c=trainset.targets.numpy(), cmap='tab10', s=1)
# plt.colorbar()
# plt.title('Embedded MNIST Handwritten Digits')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.show()

# Run probabilistic PCA on the neighborhood of 500 points to determine the effective dimension
from sklearn.decomposition import PCA
from utils import compute_effective_dimension
import numpy as np
# import pynndescent
import matplotlib.pyplot as plt
import tqdm

import warnings

warnings.filterwarnings("ignore")

dim = 784
max_dim = 300
num_points, k = 100, 500
print(f"Number of points: {num_points}, Number of neighbors: {k}, Dimension of data: {dim}")
# index = pynndescent.NNDescent(data, n_neighbors=k)
test_points_idx = np.random.choice(data.shape[0], num_points, replace=False).astype(int) # randomly select 100 points from the dataset and convert to integer scalar array
# test_points_neighbors = index.neighbor_graph[0][test_points_idx] # find the K nearest neighbors of the test points
# find the K nearest neighbors of the test points
test_points_neighbors = np.zeros((num_points, k), dtype=int)
progress = tqdm.tqdm(range(num_points), desc='Finding kNN')
for i in progress:
    test_point = data[test_points_idx[i]]
    distances = np.linalg.norm(data - test_point, axis=1)
    test_points_neighbors[i] = np.argsort(distances)[1:k+1]
print(test_points_neighbors.shape)

dimension_lst = []
visited = {}
# Perform probabilistic PCA on the tangent space of the test points to find the dimension of tangent space
progress = tqdm.tqdm(range(num_points), desc='Computing effective dimension')
for i in progress:
    test_point = data[test_points_idx[i]]
    tangent_space = data[test_points_neighbors[i]] - test_point
    # standardize the tangent space
    tangent_space = tangent_space / (np.std(tangent_space, axis=0) + 1e-7)
    
    # Probabilistic PCA
    effective_dim = compute_effective_dimension(tangent_space, max_dim, random_sample=10)
    # pca_cores = compute_scores(tangent_space, dim)
    # effective_dim = np.argmax(pca_cores) + 1

    # if effective_dim not in visited:
    #     visited[effective_dim] = 1
    #     pca_model = PCA(n_components=effective_dim)
    #     pca_model.fit(tangent_space)

    dimension_lst.append(effective_dim)

    progress.set_postfix({'Effective Dimension': effective_dim})

# Plot histogram for effective dimensions
print(dimension_lst)
print(sum(dimension_lst) / len(dimension_lst))
plt.hist(dimension_lst, bins=range(1, dim + 2), color='blue', edgecolor='black')
plt.xlabel('Effective Dimension')
plt.ylabel('Frequency')
plt.title('Histogram of Effective Dimension of Tangent Space')
plt.show()