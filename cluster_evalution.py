import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import matplotlib.pyplot as plt

# Generating synthetic data
n_samples = 1000
n_features = 2
n_clusters = 3
X, true_labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                            cluster_std=0.7, random_state=42)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(X)

# Internal Evaluation
silhouette = silhouette_score(X, predicted_labels)
calinski_harabasz = calinski_harabasz_score(X, predicted_labels)
davies_bouldin = davies_bouldin_score(X, predicted_labels)

print("Internal Evaluation Metrics:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# External Evaluation
ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
v_measure = v_measure_score(true_labels, predicted_labels)

print("\nExternal Evaluation Metrics:")
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"V-measure: {v_measure:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
plt.title("True Labels")

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
plt.title("K-means Clustering")

plt.tight_layout()
plt.show()

# Experimenting with different numbers of clusters
n_clusters_range = range(2, 7)
silhouette_scores = []
ari_scores = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    silhouette_scores.append(silhouette_score(X, labels))
    ari_scores.append(adjusted_rand_score(true_labels, labels))

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.subplot(122)
plt.plot(n_clusters_range, ari_scores, marker='o')
plt.title("Adjusted Rand Index vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Adjusted Rand Index")

plt.tight_layout()
plt.show()