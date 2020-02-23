from typing import Mapping, Tuple, List, Callable, Optional, Union

import tensorflow_probability as tfp

from src.commons.imports import tf
from src.commons.python import fill
from src.type.core import Along

TENSOR_PAIR = Tuple[tf.Tensor, tf.Tensor]
TENSOR_FUNCTION = Callable[[tf.Tensor], tf.Tensor]
AXIS_TYPE = Union[int, Optional[List[int]]]

NAN = tf.math.log(-1.0)


def bernoulli(shape: List[int], truth_probability: float = 0.5) -> tf.Tensor:
    distribution = tfp.distributions.Bernoulli(probs=truth_probability)
    return tf.cast(distribution.sample(shape), dtype=tf.dtypes.bool)


def tensor_dot(x: Along, y: Along) -> tf.Tensor:
    return tf.tensordot(x.tensor, y.tensor, axes=(x.axes, y.axes))


def slice_along(x: tf.Tensor, ranges: Mapping[int, Tuple[int, int]]) -> tf.Tensor:
    begin = fill(len(x.shape), 0)
    size = list(x.shape)
    for axis, _range in ranges:
        begin[axis] = _range[0]
        size[axis] = _range[1] - _range[0]
    return tf.slice(x, begin, size)


def broadcast_along(x: tf.Tensor, shape: List[int], axes: List[int]) -> tf.Tensor:
    reshape_shape = shape
    j = 0
    k = 0
    for i in range(len(shape)):
        if i == axes[j]:
            reshape_shape[i] = 1
            j += 1
        else:
            assert reshape_shape[i] == x.shape[k]
            k += 1

    x = tf.reshape(x, reshape_shape)
    return tf.broadcast_to(x, shape)


def cast(dtype: tf.dtypes.DType) -> TENSOR_FUNCTION:
    return lambda tensor: tf.cast(tensor, dtype)


def check_shape(tensor: tf.Tensor, shape: List[int]):  # TODO allow for None in the shape
    if not tensor.shape == shape:
        raise AssertionError(str(tensor.shape) + " != " + str(shape))


def tile_tensor_at_zeroth_dimension(tensor: tf.Tensor, count: int) -> tf.Tensor:
    multiples = tf.squeeze(
        tf.one_hot(indices=[0], depth=len(tensor.shape), on_value=count, off_value=1, dtype=tf.int32))
    return tf.tile(tensor, multiples)


def sub_tensor(tensor: tf.Tensor, axis: int, index: int) -> tf.Tensor:
    """
    Get the slice of @tensor at @index along @axis.
    
    :param tensor: the tensor from which to get the slice
    :param axis: the axis orthogonal to the slice
    :param index: the index from where the slice is to be taken
    :return: the specified slice
    """

    begin = tf.one_hot(axis, len(tensor.shape), on_value=index, dtype=tf.int32)
    size = tf.one_hot(axis, len(tensor.shape), off_value=-1, dtype=tf.int32)
    result = tf.squeeze(tf.slice(tensor, begin, size), [axis])

    result_shape = tensor.shape.as_list()
    result_shape.pop(axis)
    result.set_shape(result_shape)

    return result


def split_head(seq: tf.Tensor, axis: int) -> TENSOR_PAIR:
    """
    Split seq at the head.
    :param seq: sequence to be split
    :param axis: along which to split seq
    :return: head of seq, seq without head
    :rtype: (tf.Tensor, tf.Tensor)
    """
    head, tail = tf.split(seq, [1, -1], axis)
    squeezed_head = tf.squeeze(head, axis)
    return squeezed_head, tail


def split_feet(seq: tf.Tensor, axis: int) -> TENSOR_PAIR:
    """
    Split seq at the feet.
    :param seq: sequence to be split
    :param axis: along which to split seq
    :return: seq without feet, feet of seq
    """
    body, feet = tf.split(seq, [-1, 1], axis)
    squeezed_feet = tf.squeeze(feet, axis)
    return body, squeezed_feet


def additively_normalized(vector_batch: tf.Tensor) -> tf.Tensor:
    """
    Shifts the given vectors such that the sum of each is equal to 1.

    :param vector_batch: Batch of vectors
    :return: Batch of normalized vectors
    """
    vector_batch = tf.transpose(a=vector_batch)
    vector_length = vector_batch.shape.as_list()[0]
    vector_mean = tf.reduce_mean(input_tensor=vector_batch, axis=0)
    shift = vector_mean - 1 / vector_length
    normalized_vector_batch = vector_batch - shift
    return tf.transpose(a=normalized_vector_batch)


def multiplicatively_normalized(vector_batch: tf.Tensor) -> tf.Tensor:
    """
    Rescales the given vectors such that the sum of each is equal to 1.

    :param vector_batch: Batch of vectors
    :return: Batch of normalized vectors
    """
    vector_batch = tf.transpose(a=vector_batch)
    scale = tf.reduce_sum(input_tensor=vector_batch, axis=0)
    normalized_vector_batch = vector_batch / scale
    return tf.transpose(a=normalized_vector_batch)


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


def has_nan(tensor: tf.Tensor, axis: AXIS_TYPE = None) -> tf.Tensor:
    return tf.reduce_any(tf.math.is_nan(tensor), axis)


def nans_like(tensor: tf.Tensor) -> tf.Tensor:
    return nans(tensor.shape, tensor.dtype)


def nans(shape: List[int], dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.broadcast_to(tf.cast(NAN, dtype), shape)


def with_noise(noise: tf.Tensor, noise_proportion: float) -> TENSOR_FUNCTION:
    assert tf.rank(noise) == 0

    def closure(t: tf.Tensor) -> tf.Tensor:
        mask = bernoulli(t.shape, noise_proportion)
        broadcast_noise = tf.broadcast_to(noise, t.shape)
        return tf.where(mask, broadcast_noise, t)

    return closure
