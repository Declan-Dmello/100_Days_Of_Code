import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage

np.random.seed(42)

n_samples = 100

X = np.zeros((n_samples, 2))
X[:,0] = np.random.normal(0,1,n_samples)
X[:,1] = np.random.normal(0,5,n_samples)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 3
clustering  = AgglomerativeClustering(n_clusters=n_clusters)
clusters =clustering.fit_predict(X_scaled)

def plot_clusters_and_dendrogram(X, labels):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax1.set_title('Hierarchical Clustering Results')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    plt.colorbar(scatter)

    ax2 = fig.add_subplot(122)
    linkage_matrix = linkage(X, method='ward')
    dendrogram(linkage_matrix)
    ax2.set_title('Dendrogram')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Distance')

    plt.tight_layout()
    return plt

plot = plot_clusters_and_dendrogram(X_scaled, clusters)

for i in range(n_clusters):
    cluster_size = np.sum(clusters == i)
    cluster_data = X_scaled[clusters == i]
    cluster_mean = np.mean(cluster_data, axis=0)
    cluster_std = np.std(cluster_data, axis=0)

    print(f"\nCluster {i}:")
    print(f"Size: {cluster_size}")
    print(f"Mean: Feature1={cluster_mean[0]:.2f}, Feature2={cluster_mean[1]:.2f}")
    print(f"Std: Feature1={cluster_std[0]:.2f}, Feature2={cluster_std[1]:.2f}")

plt.show()




