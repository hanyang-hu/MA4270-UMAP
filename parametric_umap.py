import torch
import pynndescent
import numpy as np

import utils


"""
Implementation of Parametric UMAP in PyTorch.
Hyperparameters:
    1. min_dist: float, default=0.1
        The minimum effetive distance between embedded points.
    2. spread: float, default=1.0
        The scale of distance (for computing the grade of membership) between embedded points.
"""
class ParametricUMAP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, dataset, encoder, decoder, min_dist=0.1, spread=1.0):
        super(ParametricUMAP, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.encoder = encoder
        self.decoder = decoder

        self.dataset = dataset
        assert dataset.shape[1] == input_dim, "Input dimension mismatch."

        self.a, self.b = utils.find_ab_params(spread, min_dist)

    def forward(self, x):
        return self.encoder(x)
    
    def reconstruct(self, y):
        return self.decoder(y)
    
    """
    Construct the fuzzy simplicial set to approximate the manifold.
    Return the umap fraph, the rhos and the sigmas.
    Hyperparameters:
        1. K: int, default=15
            The number of nearest neighbors to consider.
    """
    def construct_fuzzy_simplicial_set(self, K=15):
        index = pynndescent.NNDescent(self.dataset, n_neighbors=K+1)
        neighbor_graph = index.neighbor_graph[0][:,1:]
        neighbor_distances = index.neighbor_graph[1][:,1:]
        
        rhos, sigmas = utils.get_rhos_and_sigmas(neighbor_distances) # only the distances are needed
    
    """
    Train the encoder and decoder through stochastic gradient descent.
    Hyperparameters:
        1. epochs: int, default=100
            The number of epochs to train the model.
        2. batch_size: int, default=128
            The number of samples in each batch.
        3. lr: float, default=1e-3
            The learning rate for the optimizer.
        4. pearson: float, default=0.1
            The Pearson correlation coefficient for the Gaussian likelihood.
    Returns a list of losses (cross-entropy, Pearson correlation and reconstruction loss) for each epoch.
    """
    def fit(self, epochs=100, batch_size=64, lr=1e-3, pearson=0.1):
        pass


if __name__ == '__main__':
    # To test whether the code runs without error
    X = torch.randn(100, 10)
    encoder = utils.MLP(10, [64, 32], 2)
    decoder = utils.MLP(2, [32, 64], 10)
    model = ParametricUMAP(10, 2, X, encoder, decoder)
    model.construct_fuzzy_simplicial_set()
