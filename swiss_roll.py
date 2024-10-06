from BayesianPCA import bayespca
import pynndescent
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import matplotlib.pyplot as plt

# Construct a 3D Swiss roll dataset
dataset = make_swiss_roll(n_samples=2000, noise=0.2, hole=True)[0]
data = torch.Tensor(dataset)
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
# Perform Bayesian PCA on the tangent space on the test points to find dimension of tangent space
for i in range(num_points):
    test_point = data[test_points_idx[i]]
    tangent_space = data[test_points_neighbors[i]] - test_point
    b = bayespca.BayesPCA()
    b.fit_transfrom(tangent_space)
    dimension_lst.append(b.n_components_)
    print(f"Test point {i+1}: Dimension of tangent space = {bayespca.n_components_}")
