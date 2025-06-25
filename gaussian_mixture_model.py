import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n_samples = 300

cluster1 = np.random.normal(loc=[0, 2], scale=0.5, size=(n_samples//2, 2))
cluster2 = np.random.normal(loc=[3, 0], scale=1.0, size=(n_samples//2, 2))

X = np.vstack([cluster1, cluster2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set1',alpha=0.5)
plt.title('Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(132)
for i in range(2):
    mask = labels == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],label=f'Cluster {i+1}', alpha=0.5)
    plt.scatter(gmm.means_[i, 0], gmm.means_[i, 1],c='black', marker='x', s=100,linewidth=3, label=f'Center {i+1}')
plt.title('Clusters with Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(133)
probabilities = gmm.predict_proba(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],c=probabilities[:, 0], cmap='coolwarm',alpha=0.5)
plt.colorbar(label='Probability of Cluster 1')
plt.title('Membership Probabilities')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print(f"Number of clusters: {gmm.n_components}")
for i, weight in enumerate(gmm.weights_):
    print(f"Cluster {i+1}: {weight:.3f}")

for i in range(5):
    probs = gmm.predict_proba(X_scaled[i:i+1])
    print(f"\nPoint {i+1}:")
    print(f"Probability it belongs to cluster 1 : {probs[0,0]:.3f}")
    print(f"Probability it belongs to cluster 2 : {probs[0,1]:.3f}")