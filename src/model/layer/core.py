from src.commons.imports import tf


def double_relu(x, feature_axis: int):
    """
    See "Understanding and Improving Convolutional Neural Networks
    via Concatenated Rectified Linear Units".
    """
    return tf.concat([tf.nn.relu(x), tf.nn.relu(-x)], axis=feature_axis)
