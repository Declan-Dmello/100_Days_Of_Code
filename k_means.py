import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def euclidean_distance(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KMeans:
    def __init__(self,  K = 5 , max_iters = 100, plot_steps= False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        #list of sample indices
        self.clusters = [[] for _ in range(self.K)]

        #store centers (mean, vector for each cluster
        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples , self.n_features = X.shape

        #initializa the centroids
        random_sample_indxs = np.random.choice(self.n_samples, self.K, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_indxs]

        #optimizing the cluster

        for _ in range(self.max_iters):
            #assigning samples to closest samples
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroid_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()


        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self,clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx ,cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx




    def _create_clusters(self,centroids):
        #assigning sample to closest cluster
        clusters = [[] for _ in range(self.K)]
        for idx , sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters


    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx



    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis =0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroid_old, centroids):
        distances = [euclidean_distance(centroid_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0




    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))
        for i , index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)




if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150 , plot_steps=True)
    y_pred = k.predict(X)


    print("Internal Cluster Evaluation")
    silhouette = silhouette_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X,y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)

    print("Internal Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")






