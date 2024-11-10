from parametric_umap import ParametricUMAP
from utils import MLP
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
print("Number of samples in the MNIST dataset:", len(trainset))

# Convert the dataset to a numpy array of shape (60000, 784)
data = trainset.data.numpy().reshape(-1, 784)
# Normalize the data to have zero mean and unit variance
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-7)

# Define the model
embed_dim = 200
encoder = MLP(784, [1024, 2048, 1024, 512], embed_dim).to(torch.device('cuda'))
model = ParametricUMAP(784, 2, data, encoder, K=15).to(torch.device('cuda'))
try:
    model.load_state_dict(torch.load('./mnist_model.pth', weights_only=True))
    print("Loaded the model from mnist_model.pth")
except:
    print("Failed to load mnist_model.pth")

# Train the model
ce_loss, pearson_loss = model.fit(epochs=500, batch_size=1024, negative_samples=5, pearson_coef=0.01)

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')

# Plot the loss curves
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
axs[0].plot(ce_loss, label="Cross-Entropy Loss")
axs[0].set_title('Cross-Entropy Loss')
axs[1].plot(pearson_loss, label="Pearson Correlation")
axs[1].set_title('Pearson Correlation')
plt.show()

# Plot the embedded MNIST dataset
embedding = model.transform(torch.tensor(data, dtype=torch.float).to(model.device)).detach().cpu().numpy()
if embed_dim == 2:
    plt.scatter(embedding[:, 0], embedding[:, 1], c=trainset.targets.numpy(), cmap='tab10', s=1)
    plt.colorbar()
    plt.title('Embedded MNIST Handwritten Digits')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()
else:
    # run pca on the embedded data and plot the first two principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(embedding)
    embedding_pca = pca.transform(embedding)
    plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=trainset.targets.numpy(), cmap='tab10', s=1)
    plt.colorbar()
    plt.title('Embedded MNIST')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

# from umap.parametric_umap import ParametricUMAP

# embedder = ParametricUMAP(n_epochs=50, verbose=True)
# embedding = embedder.fit_transform(data)

# plt.scatter(embedding[:, 0], embedding[:, 1], c=trainset.targets.numpy(), cmap='tab10')
# plt.colorbar()
# plt.title('Embedded MNIST')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.show()
