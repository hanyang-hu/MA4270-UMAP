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
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

# Convert the dataset to a numpy array of shape (60000, 784)
data = trainset.data.numpy().reshape(-1, 784)

# Define the model
encoder = MLP(784, [512, 512, 512, 64], 2)
model = ParametricUMAP(784, 2, data, encoder, K=15)

# Train the model
ce_loss, pearson_loss = model.fit(epochs=50, batch_size=512, negative_samples=5, pearson_coef=0.0)

# Plot the loss curves
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
axs[0].plot(ce_loss, label="Cross-Entropy Loss")
axs[0].set_title('Cross-Entropy Loss')
axs[1].plot(pearson_loss, label="Pearson Correlation")
axs[1].set_title('Pearson Correlation')
plt.show()

# Plot the embedded Fashion MNIST dataset
embedding = model.transform(torch.tensor(data, dtype=torch.float).to(model.device)).detach().cpu().numpy()
plt.scatter(embedding[:, 0], embedding[:, 1], c=trainset.targets.numpy())
plt.colorbar()
plt.title('Embedded Fashion MNIST')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()

