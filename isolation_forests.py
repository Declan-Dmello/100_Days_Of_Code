import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def detect_anomalies(X, contamination=0.1):

    clf = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    clf.fit(X)

    predictions = clf.predict(X)
    scores = -clf.score_samples(X)

    anomaly_mask = predictions == -1
    anomalies = X[anomaly_mask]
    anomaly_indices = np.where(anomaly_mask)[0]

    return {
        'anomalies': anomalies,
        'indices': anomaly_indices,
        'scores': scores[anomaly_mask]
    }




def visualize_anomalies(X, anomalies):

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Normal')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly')
    plt.title('Isolation Forest Anomaly Detection')
    plt.legend()
    plt.show()


np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.array([[4, 4], [-4, -4]])
X = np.vstack([normal_data, anomalies])

results = detect_anomalies(X)
print("Anomaly Indices:", results['indices'])
visualize_anomalies(X, results['anomalies'])