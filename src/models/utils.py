from tensorflow.keras import initializers
from tensorflow.keras import layers

from src.imports import tf


def dense_relu_layer(inp: tf.Tensor, width: int) -> tf.Tensor:
    x = layers.Dense(width, kernel_initializer=initializers.he_normal())(inp)
    return layers.ReLU()(x)
