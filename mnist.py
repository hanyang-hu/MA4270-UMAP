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
encoder = MLP(784, [5000, 5000, 5000, 1024, 512], 2).to(torch.device('cuda'))
model = ParametricUMAP(784, 2, data, encoder, K=15).to(torch.device('cuda'))

# Train the model
ce_loss, pearson_loss = model.fit(epochs=100, batch_size=1024, negative_samples=5, pearson_coef=0.0)

# Plot the loss curves
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
axs[0].plot(ce_loss, label="Cross-Entropy Loss")
axs[0].set_title('Cross-Entropy Loss')
axs[1].plot(pearson_loss, label="Pearson Correlation")
axs[1].set_title('Pearson Correlation')
plt.show()

# Plot the embedded MNIST dataset
embedding = model.transform(torch.tensor(data, dtype=torch.float).to(model.device)).detach().cpu().numpy()
plt.scatter(embedding[:, 0], embedding[:, 1], c=trainset.targets.numpy(), cmap='tab10')
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