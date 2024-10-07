import pynndescent
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
import torch
import matplotlib.pyplot as plt

# Construct a 3D Swiss roll dataset
dataset = make_swiss_roll(n_samples=5000, noise=0.2, hole=True)[0]
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
num_points, k = 100, 100
index = pynndescent.NNDescent(data, n_neighbors=k)
test_points_idx = np.random.choice(data.shape[0], num_points, replace=False).astype(int) # randomly select 100 points from the dataset and convert to integer scalar array
test_points_neighbors = index.neighbor_graph[0][test_points_idx] # find the K nearest neighbors of the test points

# Modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
def compute_scores(X, n_components):
    fa = FactorAnalysis()

    fa_scores = []
    for n in range(n_components):
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return fa_scores

dimension_lst = []
# Perform factor analysis on the tangent space on the test points to find dimension of tangent space
for i in range(num_points):
    test_point = data[test_points_idx[i]]
    tangent_space = data[test_points_neighbors[i]] - test_point
    # standardize the tangent space
    tangent_space = tangent_space / torch.std(tangent_space, axis=0)

    if i == 0:
        # Plot the tangent space of the first test point
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tangent_space[:, 0], tangent_space[:, 1], tangent_space[:, 2])
        plt.title('Tangent Space of Test Point 1')
        plt.show()
        
    
    # Factor analysis
    fa_cores = compute_scores(tangent_space.numpy(), dim)
    effective_dim = np.argmax(fa_cores) + 1

    dimension_lst.append(effective_dim)
    if i % 10 == 0:
        print(f'Test point {i+1}: Effective dimension of tangent space = {effective_dim}')

# Plot histogram of the effective dimension of the tangent space
plt.hist(dimension_lst, bins=20, color='blue', edgecolor='black')
plt.xlabel('Effective Dimension')
plt.ylabel('Frequency')
plt.title('Histogram of Effective Dimension of Tangent Space')
plt.show()
