import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sample_distances(D, num_samples=10000):
    distances = []
    for _ in range(num_samples):
        point1 = np.random.rand(D)
        point2 = np.random.rand(D)
        distance = np.linalg.norm(point1 - point2)
        distances.append(distance)
    return distances

dimensions = [3, 10, 50, 100, 1000]
num_samples = 10000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

for D in dimensions:
    distances = sample_distances(D, num_samples)
    
    # Plot original distances
    sns.kdeplot(distances, shade=True, label=f'D={D}', ax=ax1)
    
    # Plot rescaled distances
    rescaled_distances = [d / np.sqrt(D) for d in distances]
    sns.kdeplot(rescaled_distances, shade=True, linestyle='--', label=f'D={D}', ax=ax2)

# Set titles and labels for the subplots
ax1.set_title('Estimated Density of the Distance between Two Random Points')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Density')
# ax1.legend(title='Original Distances')

ax2.set_title('Estimated Density of the Scaled Distance between Two Random Points')
ax2.set_xlabel('Scaled Distance')
ax2.set_ylabel('Density')
# ax2.legend(title='Rescaled Distances')

# Combine legends of both subplots and place them at the bottom center
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(dimensions), title='Dimensionality', fontsize='large', title_fontsize='large')
# Set font sizes for all labels and titles
ax1.title.set_fontsize('x-large')
ax1.xaxis.label.set_fontsize('large')
ax1.yaxis.label.set_fontsize('large')
ax1.tick_params(axis='both', which='major', labelsize='medium')

ax2.title.set_fontsize('x-large')
ax2.xaxis.label.set_fontsize('large')
ax2.yaxis.label.set_fontsize('large')
ax2.tick_params(axis='both', which='major', labelsize='medium')
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the legend
plt.show()
