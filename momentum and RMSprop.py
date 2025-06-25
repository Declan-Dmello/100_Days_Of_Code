import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


def create_model(optimizer):
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(20,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


momentum_optimizer = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)

rmsprop_optimizer = keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9
)

model_momentum = create_model(momentum_optimizer)
model_rmsprop = create_model(rmsprop_optimizer)

print("Momentum")
history_momentum = model_momentum.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\n\nRMSprop")
history_rmsprop = model_rmsprop.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


_, momentum_acc = model_momentum.evaluate(X_test, y_test, verbose=0)
_, rmsprop_acc = model_rmsprop.evaluate(X_test, y_test, verbose=0)

print(f'\nAccuracy-')

print(f'Momentum: {momentum_acc:.4f}')
print(f'RMSprop:  {rmsprop_acc:.4f}')