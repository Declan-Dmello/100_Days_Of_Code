import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Generate some real data (a simple 2D Gaussian distribution)
def generate_real_samples(n):
    x1 = np.random.normal(loc=0, scale=1, size=n)
    x2 = np.random.normal(loc=0, scale=1, size=n)
    X = np.column_stack((x1, x2))
    y = np.ones((n, 1))
    return X, y



def the_generator():
    model = keras.Sequential([
        layers.Dense(4, input_dim=2),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(2)
    ])
    return model


def the_discriminator():
    model = keras.Sequential([
        layers.Dense(4, input_dim=2),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_gan(epochs=1000, batch_size=128):
    generator = the_generator()
    discriminator = the_discriminator()





    # Build combined model
    discriminator.trainable = False
    gan_input = layers.Input(shape=(2,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.Model(gan_input, gan_output)

    # Compile models
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Training loop
    for epoch in range(epochs):
        # Train discriminator
        # Real samples
        x_real, y_real = generate_real_samples(batch_size)
        # Fake samples
        noise = np.random.normal(0, 1, size=(batch_size, 2))
        x_fake = generator.predict(noise)
        y_fake = np.zeros((batch_size, 1))

        # Train discriminator on real and fake samples
        d_loss_real = discriminator.train_on_batch(x_real, y_real)
        d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, size=(batch_size, 2))
        y_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y_gan)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

            # Plot the current state
            if epoch % 500 == 0:
                plot_distributions(generator, epoch)


def plot_distributions(generator, epoch):
    # Generate real samples
    x_real, _ = generate_real_samples(1000)

    # Generate fake samples
    noise = np.random.normal(0, 1, size=(1000, 2))
    x_fake = generator.predict(noise)

    # Plot
    plt.figure(figsize=(10, 5))

    # Real samples
    plt.subplot(1, 2, 1)
    plt.scatter(x_real[:, 0], x_real[:, 1], c='blue', alpha=0.5)
    plt.title('Real Data')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Fake samples
    plt.subplot(1, 2, 2)
    plt.scatter(x_fake[:, 0], x_fake[:, 1], c='red', alpha=0.5)
    plt.title(f'Generated Data (Epoch {epoch})')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    plt.tight_layout()
    plt.show()


# Train the GAN
if __name__ == "__main__":
    train_gan(epochs=2000, batch_size=128)