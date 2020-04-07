"""VGG16 network."""
from typing import Tuple

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

VGG16_MIN_IMAGE_SIZE = (48, 48)

def image_is_smaller_than_vgg_min(image_shape):
    return image_shape[0] < VGG16_MIN_IMAGE_SIZE[0] or \
        image_shape[1] < VGG16_MIN_IMAGE_SIZE[1]


def vgg16(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """Return VGG16 Keras model."""
    num_classes = output_shape[0]
    image_input = Input(shape=input_shape)
    resized_input = image_input

    if len(input_shape) < 3:
        resized_input = Lambda(lambda x: K.repeat_elements(tf.expand_dims(x, -1), 3, -1))(resized_input)
        input_shape = (input_shape[0], input_shape[1], 3)

    if image_is_smaller_than_vgg_min(resized_input.shape):
        resized_input = Lambda(lambda x: tf.image.resize(x, VGG16_MIN_IMAGE_SIZE))(resized_input)
        input_shape = VGG16_MIN_IMAGE_SIZE + (3,)

    vgg = VGG16(include_top=False, input_shape=input_shape)
    for layer in vgg.layers:
        layer.trainable = False

    extracted_features = vgg(resized_input)
    flattened_features = Flatten()(extracted_features)
    class1 = Dense(1024, activation='relu')(flattened_features)
    output = Dense(num_classes, activation='softmax')(class1)
    model = Model(inputs=image_input, outputs=output)

    return model
