import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dim=8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = self.creating_autoencoder()


    def creating_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoder = Dense(self.input_dim, activation='linear')(encoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder

    def train(self, X_train, epochs=50, batch_size=32):

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0)

    def detect_anomalies(self, X_test, threshold_percentile=95):

        reconstructed = self.autoencoder.predict(X_test)

        mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

        threshold = np.percentile(mse, threshold_percentile)
        anomaly_mask = mse > threshold

        return {
            'anomalies': X_test[anomaly_mask],
            'anomaly_indices': np.where(anomaly_mask)[0],
            'reconstruction_error': mse,
            'threshold': threshold
        }


# Demonstration
def demonstrate_autoencoder_anomaly_detection():
    # Generate synthetic data
    np.random.seed(42)

    # Normal data
    X_normal = np.random.normal(loc=0, scale=1, size=(100, 10))

    # Anomalies
    X_anomaly = np.random.uniform(low=-5, high=5, size=(10, 10))
    X_combined = np.vstack([X_normal, X_anomaly])

    # Initialize and train detector
    detector = AutoencoderAnomalyDetector(input_dim=10)
    detector.train(X_normal)

    # Detect anomalies
    results = detector.detect_anomalies(X_combined)

    print("Anomaly Indices:", results['anomaly_indices'])
    print("Number of Anomalies:", len(results['anomalies']))

# Uncomment to run demonstration
demonstrate_autoencoder_anomaly_detection()