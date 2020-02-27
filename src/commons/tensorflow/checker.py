from typing import List

from src.commons.tensorflow.typealias import AXIS_TYPE

from src.commons.imports import tf


def has_nan(tensor: tf.Tensor, axis: AXIS_TYPE = None) -> tf.Tensor:
    return tf.reduce_any(tf.math.is_nan(tensor), axis)


def check_shape(tensor: tf.Tensor, shape: List[int]):  # TODO allow for None in the shape
    if not tensor.shape == shape:
        raise AssertionError(str(tensor.shape) + " != " + str(shape))
