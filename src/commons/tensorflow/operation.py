from src.commons.tensorflow.utils import AXIS_TYPE, TENSOR_FUNCTION

from src.commons.imports import tf
from src.commons.tensorflow.checker import has_nan
from src.commons.tensorflow.maker import nans_like
from src.type.core import Along


def tensor_dot(x: Along, y: Along) -> tf.Tensor:
    return tf.tensordot(x.tensor, y.tensor, axes=(x.axes, y.axes))


def cast(dtype: tf.dtypes.DType) -> TENSOR_FUNCTION:
    return lambda tensor: tf.cast(tensor, dtype)


def batch_average(function: TENSOR_FUNCTION, axis: AXIS_TYPE = -1) -> TENSOR_FUNCTION:
    """
    :param function: to be mapped over a batch and averaged
    :param axis: along which dimension to perform reduce_mean
    :return: a new function that works on batches
    """

    def average(batch):
        return tf.reduce_mean(input_tensor=function(batch), axis=axis)

    return average


def batch_dot_product(v_1: tf.Tensor, v_2: tf.Tensor, axis: AXIS_TYPE = -1) -> tf.Tensor:
    """
    Each vector in the batch v_1 gets paired with the corresponding vector in v_2.
    The batch axis is axis 0. So the sum is performed along axis 1.

    :param v_1: a batch of vectors
    :param v_2: another batch of vectors
    :param axis: along which dimension to perform reduce_sum
    :return: the batch of dot products
    """
    return tf.reduce_sum(input_tensor=v_1 * v_2, axis=axis)


def reduce_min(tensor: tf.Tensor, axis: AXIS_TYPE = None) -> tf.Tensor:
    """Temporary fix while tf.reduce_min is broken"""
    min_of_non_nans = tf.reduce_min(tensor, axis)
    return tf.where(has_nan(tensor, axis), nans_like(min_of_non_nans), min_of_non_nans)


def reduce_max(tensor: tf.Tensor, axis: AXIS_TYPE = None) -> tf.Tensor:
    """Temporary fix while tf.reduce_max is broken"""
    max_of_non_nans = tf.reduce_max(tensor, axis)
    return tf.where(has_nan(tensor, axis), nans_like(max_of_non_nans), max_of_non_nans)
