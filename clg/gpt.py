import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Define the generator model


def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * 7 * 7, input_dim=latent_dim, activation='relu'))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(
        2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(
        2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (7, 7), padding='same', activation='tanh'))
    return model

# Define the discriminator model


def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same',
              input_shape=img_shape, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Define the GAN model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the discriminator


def compile_discriminator(discriminator):
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])

# Compile the GAN


def compile_gan(gan):
    gan.compile(loss='binary_crossentropy', optimizer='adam')

# Generate random samples for training


def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(latent_dim * n_samples).reshape((n_samples, latent_dim))

# Generate fake samples with the generator


def generate_fake_samples(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    fake_samples = generator.predict(latent_points)
    labels = np.zeros((n_samples, 1))
    return fake_samples, labels

# Train the GAN


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs, batch_size):
    batch_per_epoch = dataset.shape[0] // batch_size

    for epoch in range(n_epochs):
        for batch in range(batch_per_epoch):
            # Train discriminator
            real_samples = dataset[batch *
                                   batch_size: (batch + 1) * batch_size]
            real_labels = np.ones((batch_size, 1))

            fake_samples, fake_labels = generate_fake_samples(
                generator, latent_dim, batch_size)

            d_loss_real = discriminator.train_on_batch(
                real_samples, real_labels)
            d_loss_fake = discriminator.train_on_batch(
                fake_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = generate_latent_points(latent_dim, batch_size)
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

            # Print progress
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Batch {batch}/{batch_per_epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")


# Example usage
latent_dim = 100
img_shape = (28, 28, 1)
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

compile_discriminator(discriminator)
compile_gan(gan)

train_gan(generator, discriminator, gan, x_train,
          latent_dim, n_epochs=10, batch_size=128)
