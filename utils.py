from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit
import numpy as np
import torch
import torch.nn.functional as F


# Modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
def compute_scores(X, n_components):
    pca = PCA(whiten=True)

    pca_scores = []
    for n in range(1, n_components+1):
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))

    return pca_scores


"""
For more efficient computation in the high-dimensional setting, we randomly sample 50 dimensions and run probabilistic PCA.
The effective dimension is the one with the highest PCA score.
"""
def compute_effective_dimension(X, dim, random_sample=50):
    random_dim = np.random.choice(np.arange(dim), random_sample, replace=False) # randomly select 50 dimensions
    pca = PCA(whiten=True)
    
    pca_scores = []
    for n in random_dim:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        # print(f"Dimension: {n}, PCA Score: {pca_scores[-1]}")

    return random_dim[np.argmax(pca_scores)].item()


# Obtained from the original UMAP implementation
# See https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1386 
def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


"""
Compute the rho and sigma values for each point in the dataset given the distances to the nearest neighbors.
Input: 
    neighbor_distances: np.array, shape (n_samples, K)
        The distances to the K nearest neighbors of each point in the
"""
def get_rhos_and_sigmas(neighbor_distances):
    K = neighbor_distances.shape[1]

    rhos = neighbor_distances[:,0] # the distance to the nearest neighbor

    sigmas = np.zeros_like(rhos)
    for i in range(len(rhos)):
        lo = 0.0
        hi = np.max(neighbor_distances[i]) - rhos[i]
        mid = (lo + hi) / 2.0

        while hi - lo > 1e-5:
            psum = np.sum(np.exp(-np.maximum(0.0, neighbor_distances[i] - rhos[i]) / mid))
            if psum > np.log2(K):
                hi = mid
            else:
                lo = mid
            mid = (lo + hi) / 2.0

        sigmas[i] = mid

    return rhos, sigmas


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        layer_dim = [input_dim,] + hidden_dim + [output_dim,]
        self.fc = torch.nn.ParameterList([torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 2)])
        self.last_layer = torch.nn.Linear(layer_dim[-2], layer_dim[-1])

        # Initialize weights and biases
        for layer in self.fc:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.last_layer.weight)
        self.last_layer.bias.data.fill_(0)
        
            
    def forward(self, x):
        for layer in self.fc:
            x = F.relu(layer(x))
        
        return self.last_layer(x)