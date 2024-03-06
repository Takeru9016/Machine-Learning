import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load Fashion MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize pixel values to between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Set the dimensions of the latent space
latent_dim = 2

# Define the encoder network
original_dim = x_train.shape[1]
inputs = tf.keras.Input(shape=(original_dim,))
h = layers.Dense(256, activation='relu')(inputs)
h = layers.Dense(128, activation='relu')(h)

# Define the mean and log-variance layers for the latent space
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

# Define the sampling layer


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Define the encoder model
encoder = models.Model(inputs, [z_mean, z_log_var, z])

# Define the decoder network
decoder_h = layers.Dense(128, activation='relu')
decoder_mean = layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Define the decoder model
decoder = models.Model(inputs, x_decoded_mean)

# Define the VAE model
x = decoder(inputs)
vae = models.Model(inputs, x)

# Flatten the input and output for mean squared error loss
# Flatten the input and output for mean squared error loss
inputs_flat = tf.keras.backend.flatten(inputs)
x_flat = tf.keras.backend.flatten(x)

print(inputs_flat.shape)
print(x_flat.shape)

# Define the VAE loss
reconstruction_loss = tf.keras.losses.mean_squared_error(inputs_flat, x_flat)
reconstruction_loss *= original_dim
kl_loss = - 0.5 * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)


# Compile the VAE model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Define training parameters
epochs = 1  # You can adjust the number of epochs
batch_size = 64  # You can adjust the batch size

# Train the VAE
vae.fit(x_train, epochs=epochs, batch_size=batch_size,
        validation_data=(x_test, None))
