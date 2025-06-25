import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=200, noise=0.15, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)



def dbscan_graph(X, clusters):
    plt.figure(figsize=(10, 6))

    core_mask = np.zeros_like(clusters, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True

    noise_mask = clusters == -1
    plt.scatter(X[noise_mask, 0], X[noise_mask, 1],
                c='black', label='Noise')

    plt.scatter(X[~noise_mask & core_mask, 0], X[~noise_mask & core_mask, 1],
                c='blue', label='Core')
    plt.scatter(X[~noise_mask & ~core_mask, 0], X[~noise_mask & ~core_mask, 1],
                c='green', label='Border')




    plt.title('Point Types in DBSCAN')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.tight_layout()
    plt.show()



dbscan_graph(X_scaled, clusters)


n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"\nNumber of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

for i in range(n_clusters):
    cluster_size = np.sum(clusters == i)
    print(f"\nCluster {i}:")
    print(f"Size: {cluster_size}")