import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# ------Uncomment this when you run it on your local machine once------
# # Load MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Preprocess the data
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Create & Train the Model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('handify.model')
# ---------------------------------------------------------------------

# Load the created model instead of re-running the model again & again
model = tf.keras.models.load_model('handify.model')

# Uncomment this to see the model summary
# # Calaculate the loss and accuracy of the model
# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

# Load the image from the local machine
image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error !")
    finally:
        image_number += 1
