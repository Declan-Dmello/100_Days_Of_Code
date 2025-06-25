import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(regularization_type=None, l1_factor=0.01, l2_factor=0.01):
    if regularization_type == 'l1':
        regularizer = regularizers.l1(l1_factor)
    elif regularization_type == 'l2':
        regularizer = regularizers.l2(l2_factor)
    elif regularization_type == 'l1_l2':
        regularizer = regularizers.l1_l2(l1=l1_factor, l2=l2_factor)
    else:
        regularizer = None

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,), kernel_regularizer=regularizer),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

models = {
    'No Regularization': create_model(),
    'L1 Regularization': create_model('l1'),
    'L2 Regularization': create_model('l2'),
    'L1 + L2 Regularization': create_model('l1_l2')
}

for name, model in models.items():
    print(f"\nTraining the{name} model:")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"{name} accuracy: {test_accuracy:.4f}")

