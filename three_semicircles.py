from parametric_umap import ParametricUMAP
from utils import MLP
import torch
import matplotlib.pyplot as plt

# Generate two circles, one inside the other, with some noise
n_samples = 500
X_inner = torch.randn(n_samples, 2) * 0.1
X_outer = torch.randn(n_samples, 2) * 0.1
theta = torch.linspace(0, 3.1416, n_samples).view(-1, 1)
X_inner[:, 0] += 1.0 * torch.cos(theta).squeeze()
X_inner[:, 1] += 1.0 * torch.sin(theta).squeeze()
X_outer[:, 0] += 3.0 * torch.cos(theta).squeeze()
X_outer[:, 1] += 3.0 * torch.sin(theta).squeeze()
X = torch.cat([X_inner, X_outer], dim=0)
colors = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)], dim=0)

# Define the model
embed_dim = 2
encoder = MLP(2, [32, 128, 128, 128, 32], embed_dim).to(torch.device('cuda'))
model = ParametricUMAP(2, 2, X, encoder, K=15).to(torch.device('cuda'))
try:
    model.load_state_dict(torch.load('./two_circles_model.pth', weights_only=True))
    print("Loaded the model from two_circles_model.pth")
except:
    print("Failed to load two_circles_model.pth")

# Train the model
ce_loss, pearson_loss = model.fit(epochs=1000, batch_size=256, negative_samples=200, pearson_coef=0.0, lr=0.001)

# Save the model
torch.save(model.state_dict(), 'two_circles_model.pth')

# Add a circle between the two circles and test the model
X_middle = torch.randn(n_samples, 2) * 0.05
X_middle[:, 0] += 1.75 * torch.cos(theta).squeeze()
X_middle[:, 1] += 1.75 * torch.sin(theta).squeeze()
X = torch.cat([X_inner, X_middle, X_outer], dim=0)
colors = torch.cat([torch.zeros(n_samples), 0.5 * torch.ones(n_samples), torch.ones(n_samples)], dim=0)

# Plot the original data and embeddings
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.scatter(X_inner[:, 0], X_inner[:, 1], c='r', s=1)
plt.scatter(X_middle[:, 0], X_middle[:, 1], c='g', s=1)
plt.scatter(X_outer[:, 0], X_outer[:, 1], c='b', s=1)
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(2, 2, 2) # plot the K-means clustering, K=3
# compute the K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=1)
plt.title('K-means Clustering of Original Data')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(2, 2, 3)
embedding = model.transform(X.to(torch.device('cuda'))).detach().cpu()
plt.scatter(embedding[:n_samples, 0], embedding[:n_samples, 1], c='r', s=1)
plt.scatter(embedding[n_samples:2*n_samples, 0], embedding[n_samples:2*n_samples, 1], c='g', s=1)
plt.scatter(embedding[2*n_samples:, 0], embedding[2*n_samples:, 1], c='b', s=1)
plt.title('Embedded Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

plt.subplot(2, 2, 4)
kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)
plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans.labels_, s=1)
plt.title('K-means Clustering of Embedded Data')
# plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()
