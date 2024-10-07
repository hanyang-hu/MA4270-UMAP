import pynndescent
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from utils import compute_scores

# Construct a 3D Swiss roll dataset
dataset = make_swiss_roll(n_samples=2000, noise=0.2, hole=True)[0]
data = torch.Tensor(dataset)
dim = data.shape[1]
colors = data[:, 2] # assign color to each point based on the 3rd dimension of the data

# Plot the dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
plt.title('Swiss Roll Dataset')
plt.show()

# Find the approximate K nearest neighbors of 100 test points in the dataset
num_points, k = 100, 50
index = pynndescent.NNDescent(data, n_neighbors=k)
test_points_idx = np.random.choice(data.shape[0], num_points, replace=False).astype(int) # randomly select 100 points from the dataset and convert to integer scalar array
test_points_neighbors = index.neighbor_graph[0][test_points_idx] # find the K nearest neighbors of the test points

dimension_lst = []
visited = {}
# Perform probabilistic PCA on the tangent space of the test points to find the dimension of tangent space
for i in range(num_points):
    test_point = data[test_points_idx[i]]
    tangent_space = data[test_points_neighbors[i]] - test_point
    # standardize the tangent space
    tangent_space = tangent_space / torch.std(tangent_space, axis=0)
    
    # Probabilistic PCA
    pca_cores = compute_scores(tangent_space.numpy(), dim)
    effective_dim = np.argmax(pca_cores) + 1

    if effective_dim not in visited:
        visited[effective_dim] = 1
        pca_model = PCA(n_components=effective_dim)
        pca_model.fit(tangent_space)
        estimated_tangent_plane = pca_model.components_
        # Plot the tangent space and fitted PCA
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tangent_space[:, 0], tangent_space[:, 1], tangent_space[:, 2], c='blue')
        for j in range(effective_dim):
            ax.quiver(0, 0, 0, estimated_tangent_plane[j, 0], estimated_tangent_plane[j, 1], estimated_tangent_plane[j, 2])
        plt.title(f'Tangent Space identified with effective dimension = {effective_dim}')
        plt.show()

    dimension_lst.append(effective_dim)
    if i % 10 == 0:
        print(f'Test point {i+1}: Effective dimension of tangent space = {effective_dim}')

# Plot histogram for effective dimensions
print(dimension_lst)
print(sum(dimension_lst) / len(dimension_lst))
plt.hist(dimension_lst, bins=range(1, dim + 2), color='blue', edgecolor='black')
plt.xlabel('Effective Dimension')
plt.ylabel('Frequency')
plt.title('Histogram of Effective Dimension of Tangent Space')
plt.show()
