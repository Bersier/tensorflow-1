from tensorflow.keras import initializers, layers

from src.commons.imports import tf


def double_relu_layer(inp: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """
    See "Understanding and Improving Convolutional Neural Networks
    via Concatenated Rectified Linear Units".
    """
    negated_inp = layers.Lambda(lambda x: -x)(inp)
    return layers.concatenate(inputs=[layers.ReLU()(inp), layers.ReLU()(negated_inp)], axis=axis)


def dense_relu_layer(inp: tf.Tensor, width: int) -> tf.Tensor:
    x = layers.Dense(width, kernel_initializer=initializers.he_normal())(inp)
    return layers.ReLU()(x)
