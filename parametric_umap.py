import torch
import pynndescent
import numpy as np
from scipy.sparse import csr_matrix
import tqdm

import utils


"""
Implementation of Parametric UMAP in PyTorch.
Hyperparameters:
    1. min_dist: float, default=0.1
        The minimum effetive distance between embedded points.
    2. spread: float, default=1.0
        The scale of distance (for computing the grade of membership) between embedded points.
    3. K: int, default=15
        The number of nearest neighbors to consider.
"""
class ParametricUMAP(torch.nn.Module):
    def __init__(
            self, input_dim, feature_dim, dataset, encoder, min_dist=0.1, spread=1.0, 
            K=15, device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(ParametricUMAP, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = encoder.to(self.device)

        self.dataset = dataset
        assert dataset.shape[1] == input_dim, "Input dimension mismatch."

        self.a, self.b = utils.find_ab_params(spread, min_dist)
        self.K = K

        self.umap_graph, self.rhos, self.sigmas = self.construct_fuzzy_simplicial_set()
        """ 
        # Alternatively, use the UMAP implementation to construct the fuzzy simplicial set
        from umap.umap_ import fuzzy_simplicial_set
        from sklearn.utils import check_random_state
        index = pynndescent.NNDescent(self.dataset, n_neighbors=self.K)
        knn_indices = index.neighbor_graph[0]
        knn_dists = index.neighbor_graph[1]
        self.umap_graph, self.rhos, self.sigmas = fuzzy_simplicial_set(
            X = self.dataset,
            n_neighbors = self.K,
            random_state = check_random_state(None),
            metric='euclidean',
            knn_indices = knn_indices,
            knn_dists = knn_dists,
        )
        """

    def transform(self, x):
        return self.encoder(x)
    
    """
    This function is used to reconstruct the original data from the embedded data using a decoder.
    Not used in this project.
    """
    def reconstruct(self, y):
        raise NotImplementedError
    
    """
    Construct the fuzzy simplicial set to approximate the manifold.
    Return the umap graph (represented in a sparse matrix), the rhos and the sigmas.
    """
    def construct_fuzzy_simplicial_set(self):
        index = pynndescent.NNDescent(self.dataset, n_neighbors=self.K+1)
        neighbor_graph = index.neighbor_graph[0][:,1:]
        neighbor_distances = index.neighbor_graph[1][:,1:]
        
        rhos, sigmas = utils.get_rhos_and_sigmas(neighbor_distances) # only the distances are needed
    
        # construct the sparse weighted adjacency matrix using scipy.csr_matrix
        # sparse_matrix[row_ind[k], col_ind[k]] = data[k]
        data = np.exp(-np.maximum(0.0, neighbor_distances - rhos[:,None]) / sigmas[:,None]).flatten()
        row_ind = np.repeat(np.arange(self.dataset.shape[0]), self.K)
        col_ind = neighbor_graph.flatten()
        sparse_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(self.dataset.shape[0], self.dataset.shape[0])
        )

        # compute the umap graph
        umap_graph = sparse_matrix + sparse_matrix.T - sparse_matrix.multiply(sparse_matrix.T)

        return umap_graph, rhos, sigmas

    """
    Train the encoder and decoder through stochastic gradient descent.
    Hyperparameters:
        1. epochs: int, default=100
            The number of epochs to train the model.
        2. batch_size: int, default=64
            The number of samples in each batch.
        3. lr: float, default=1e-3
            The learning rate for the optimizer.
        4. pearson_coef: float, default=0.1
            The Pearson correlation coefficient for the Gaussian likelihood.
        5. negative_samples: int, default=5
            The number of negative samples for the reconstruction loss.
    Returns a list of losses (cross-entropy, Pearson correlation and reconstruction loss) for each epoch.
    """
    def fit(self, epochs=100, batch_size=64, lr=1e-2, pearson_coef=0.0, negative_samples=5):
        ce_loss, pearson_loss = [], []
        X = torch.tensor(self.dataset, dtype=torch.float)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # construct the batched edge dataset, sampled by edge weight
        row_ind, col_ind = self.umap_graph.nonzero()
        edge_weight = np.array([self.umap_graph[row_ind[i], col_ind[i]] for i in range(row_ind.shape[0])])
        edge_sampler = torch.utils.data.WeightedRandomSampler(edge_weight, len(edge_weight))
        X_ind = X[row_ind]
        X_col = X[col_ind]
        edge_dataset = torch.utils.data.TensorDataset(
            X_ind, X_col
        )
        edge_loader = torch.utils.data.DataLoader(edge_dataset, batch_size=batch_size, sampler=edge_sampler)

        progress_bar = tqdm.tqdm(range(epochs), desc="Training")
        for _ in progress_bar:
            epoch_ce_loss, epoch_pearson_loss = [], []
            for X_row, X_col in edge_loader:
                edge_num = X_row.shape[0] # the number of edges in the batch

                X_row = X_row.to(self.device)
                X_col = X_col.to(self.device)

                Z_row = self.encoder(X_row)
                Z_col = self.encoder(X_col)
                Z_row_repeated = Z_row.repeat(negative_samples, 1)
                Z_col_repeated = Z_col.repeat(negative_samples, 1)
                # shuffle the batch to get negative samples
                Z_neg = Z_col_repeated[torch.randperm(negative_samples * edge_num)]

                # compute the cross-entropy loss
                distance_embedding = torch.concat(
                    [
                        torch.norm(Z_row - Z_col, dim=1),
                        torch.norm(Z_row_repeated - Z_neg, dim=1)
                    ]
                ).to(self.device)
                probabilities_distance = 1.0 / (1.0 + self.a * distance_embedding ** (2 * self.b))
                probabilieis_graph = torch.concat(
                    [
                        torch.ones(edge_num),
                        torch.zeros(edge_num * negative_samples)
                    ]
                ).to(self.device)
                attraction_term = -probabilieis_graph * torch.log(torch.clamp(probabilities_distance, 1e-4, 1.0))
                repulsion_term = -(1.0 - probabilieis_graph) * torch.log(torch.clamp(1 - probabilities_distance, 1e-4, 1.0))
                cross_entropy_loss = torch.mean(attraction_term + repulsion_term)
                epoch_ce_loss.append(cross_entropy_loss.item())

                # compute the covariance between distances in X[row_ind] and Z_row
                if False:
                    pearson_corr = 0.0
                    epoch_pearson_loss.append(0.0)
                else:
                    X_row_dist = torch.cdist(X_row, X_row, p=2).flatten()
                    Z_row_dist = torch.cdist(Z_row, Z_row, p=2).flatten()
                    pearson_corr = -torch.mean((X_row_dist - X_row_dist.mean()) * (Z_row_dist - Z_row_dist.mean())) / (X_row_dist.std() * Z_row_dist.std() + 1e-7)
                    epoch_pearson_loss.append(pearson_corr.item())

                # compute the total loss
                optimizer.zero_grad()
                loss = cross_entropy_loss + pearson_coef * pearson_corr
                loss.backward()
                optimizer.step()

            ce_loss.append(np.mean(epoch_ce_loss))
            pearson_loss.append(np.mean(epoch_pearson_loss))
            progress_bar.set_postfix({
                "Cross-Entropy": ce_loss[-1],
                "Pearson Corr": pearson_loss[-1],
                "Total": ce_loss[-1] + pearson_coef * pearson_loss[-1]
            })  
        
        return ce_loss, pearson_loss


if __name__ == '__main__':
    # To test whether the code runs without error
    X = torch.randn(100, 10)
    encoder = utils.MLP(10, [64, 32], 2)

    model = ParametricUMAP(10, 2, X.numpy(), encoder)
    ce_loss, pearson_loss = model.fit(epochs=100)


    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

    axs[0].plot(ce_loss, label="Cross-Entropy Loss")
    axs[0].set_title("Cross-Entropy Loss")
    axs[0].legend()

    axs[1].plot(pearson_loss, label="Pearson Correlation Loss")
    axs[1].set_title("Pearson Correlation Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    