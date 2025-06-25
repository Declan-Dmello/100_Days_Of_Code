import tensorflow as tf
from sklearn.datasets import make_classification
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
adam_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)


test_loss, test_accuracy = model.evaluate(X, y, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")


predictions = model.predict(X)


