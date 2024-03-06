import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess MNIST data
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize pixel values to the range [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=60000).batch(64)

# Generator Model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(1, (7, 7), activation='tanh', padding='same'))
    return model

# Discriminator Model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Compile Discriminator
def compile_discriminator(discriminator):
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=['accuracy']
    )

# Build GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False  # To freeze the discriminator during GAN training
    gan_input = tf.keras.Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return gan

# Training Loop
def train_gan(generator, discriminator, gan, train_dataset, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        for batch in train_dataset:
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            generated_images = generator.predict(noise)
            real_images = batch.numpy()

            # Label real and generated images
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Set parameters
latent_dim = 100
img_shape = (28, 28, 1)
epochs = 10
batch_size = 64

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
compile_discriminator(discriminator)
gan = build_gan(generator, discriminator)

# Train GAN
train_gan(generator, discriminator, gan, train_dataset, epochs, batch_size, latent_dim)
