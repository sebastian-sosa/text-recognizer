"""LeNet network."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def simple_cnn(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """Return a small CNN Keras model for character recognition."""
    num_classes = output_shape[0]

    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding='same',))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    return model
