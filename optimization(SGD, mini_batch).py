import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# 1. Generate sample dataset
np.random.seed(42)
n_samples = 10000

X = np.random.normal(0, 1, (n_samples, 20))  # 20 features
# Create a non-linear relationship
y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] ** 3 + np.random.normal(0, 0.1, n_samples)) > 0
y = y.astype(int)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



def create_model(optimizer):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate(X_train, y_train, X_test, y_test, batch_size, epochs, optimizer_name, learning_rate=0.01):

    if optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model = create_model(optimizer)
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return history.history, training_time, test_accuracy



epochs = 20
learning_rate = 0.01

print("\nBatch Gradient Descent...")
bgd_history, bgd_time, bgd_accuracy = train_and_evaluate(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    batch_size=len(X_train),
    epochs=epochs,
    optimizer_name='sgd'
)

# Stochastic Gradient Descent (batch_size=1)
print("\nStochastic Gradient Descent...")
sgd_history, sgd_time, sgd_accuracy = train_and_evaluate(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    batch_size=1,
    epochs=epochs,
    optimizer_name='sgd'
)

# Mini-batch Gradient Descent
print("\nMini-batch Gradient Descent...")
mini_history, mini_time, mini_accuracy = train_and_evaluate(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    batch_size=32,  # Mini-batch
    epochs=epochs,
    optimizer_name='sgd'
)

# 6. Plotting results
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(bgd_history['loss'], label='Batch GD')
plt.plot(sgd_history['loss'], label='SGD')
plt.plot(mini_history['loss'], label='Mini-batch GD')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(bgd_history['val_accuracy'], label='Batch GD')
plt.plot(sgd_history['val_accuracy'], label='SGD')
plt.plot(mini_history['val_accuracy'], label='Mini-batch GD')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Batch Gradient Descent")
print(f"Train time: {bgd_time:.2f} secs")
print(f"Test acc: {bgd_accuracy:.4f}")

print(f"\nStochastic Gradient Descent:")
print(f"rain time: {sgd_time:.2f} secs")
print(f"Test acc: {sgd_accuracy:.4f}")

print(f"\nMini-batch Gradient Descent:")
print(f"Train time: {mini_time:.2f} secs")
print(f"Test acc: {mini_accuracy:.4f}")









# 8. Additional function to demonstrate learning rate impact
def compare_learning_rates(X_train, y_train, X_test, y_test, batch_size=32, epochs=10):
    learning_rates = [0.1, 0.01, 0.001]
    histories = []

    plt.figure(figsize=(15, 5))

    for lr in learning_rates:
        history, _, _ = train_and_evaluate(
            X_train, y_train,
            X_test, y_test,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_name='sgd',
            learning_rate=lr
        )
        histories.append(history)

        plt.plot(history['loss'], label=f'LR = {lr}')

    plt.title('Impact of Learning Rate on Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Compare learning rates for mini-batch GD
print("\nComparing different learning rates for Mini-batch GD...")
compare_learning_rates(X_train_scaled, y_train, X_test_scaled, y_test)