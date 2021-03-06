from typing import List

from src.commons.imports import tf
from src.commons.tensorflow.typealias import AXIS_TYPE


def has_nan(tensor: tf.Tensor, axis: AXIS_TYPE = None) -> tf.Tensor:
    return tf.reduce_any(tf.math.is_nan(tensor), axis)


def check_shape(tensor: tf.Tensor, shape: List[int]):
    if not tensor.shape == shape:
        raise AssertionError(str(tensor.shape) + " != " + str(shape))
