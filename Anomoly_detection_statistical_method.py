import numpy as np
import pandas as pd


class AnomalyDetector:
    def z_score_method(data, threshold=3):
        data = np.array(data)

        mean = np.mean(data)
        std = np.std(data)

        z_scores = np.abs((data - mean) / std)

        anomalies = data[z_scores > threshold]
        anomaly_indices = np.where(z_scores > threshold)[0]

        return anomalies, anomaly_indices, z_scores

    def iqr_method(data, multiplier=1.5):
        data = np.array(data)

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        anomalies = data[(data < lower_bound) | (data > upper_bound)]
        anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

        return anomalies, anomaly_indices, lower_bound, upper_bound





def demonstrate_anomaly_detection():
    np.random.seed(42)
    normal_data = np.random.normal(loc=100, scale=15, size=1000)
    anomalies = np.array([10, 250, 300])
    data = np.concatenate([normal_data, anomalies])

    print("Anomaly Detection Demonstration:")

    print("\nZ-Score Method:")
    z_anomalies, z_indices, z_scores = AnomalyDetector.z_score_method(data)
    print(f"Detected Anomalies: {z_anomalies}")
    print(f"Anomaly Indices: {z_indices}")

    print("\nIQR Method:")
    iqr_anomalies, iqr_indices, lower_bound, upper_bound = AnomalyDetector.iqr_method(data)
    print(f"Detected Anomalies: {iqr_anomalies}")
    print(f"Anomaly Indices: {iqr_indices}")



demonstrate_anomaly_detection()