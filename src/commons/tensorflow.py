from typing import Mapping, Tuple, List

import tensorflow_probability as tfp

from src.commons.imports import tf
from src.commons.python import fill

NAN = tf.math.log(-1.0)


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


def cast(t):
    return lambda tensor: tf.cast(tensor, dtype=t)


def check_shape(tensor, shape: tuple):  # TODO allow for None in the shape
    if not tensor.shape == shape:
        raise AssertionError(str(tensor.shape) + " != " + str(shape))


def tile_tensor_at_zeroth_dimension(tensor, count):
    multiples = tf.squeeze(
        tf.one_hot(indices=[0], depth=len(tensor.shape), on_value=count, off_value=1, dtype=tf.int32))
    return tf.tile(tensor, multiples)


def sub_tensor(tensor, axis: int, index: int):
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


def split_head(seq, axis: int) -> tuple:
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


def split_feet(seq, axis: int) -> tuple:
    """
    Split seq at the feet.
    :param seq: sequence to be split
    :param axis: along which to split seq
    :return: seq without feet, feet of seq
    """
    body, feet = tf.split(seq, [-1, 1], axis)
    squeezed_feet = tf.squeeze(feet, axis)
    return body, squeezed_feet


def additively_normalized(vector_batch):
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


def multiplicatively_normalized(vector_batch):
    """
    Rescales the given vectors such that the sum of each is equal to 1.

    :param vector_batch: Batch of vectors
    :return: Batch of normalized vectors
    """
    vector_batch = tf.transpose(a=vector_batch)
    scale = tf.reduce_sum(input_tensor=vector_batch, axis=0)
    normalized_vector_batch = vector_batch / scale
    return tf.transpose(a=normalized_vector_batch)


def batch_average(function, axis=-1):
    """
    :param function: to be mapped over a batch and averaged
    :param axis: along which dimension to perform reduce_mean
    :return: a new function that works on batches
    """

    def average(batch):
        return tf.reduce_mean(input_tensor=function(batch), axis=axis)

    return average


def batch_dot_product(v_1, v_2, axis=-1):
    """
    Each vector in the batch v_1 gets paired with the corresponding vector in v_2.
    The batch axis is axis 0. So the sum is performed along axis 1.

    :param v_1: a batch of vectors
    :param v_2: another batch of vectors
    :param axis: along which dimension to perform reduce_sum
    :return: the batch of dot products
    """
    return tf.reduce_sum(input_tensor=v_1 * v_2, axis=axis)


def transposed_mean_across_examples(inputs):
    """
    :param inputs: tensor with shape (example_count, stock_count, feature_count)
    :return: mean.shape = (feature_count, stock_count)
    """
    return tf.transpose(tf.reduce_mean(inputs, axis=[0]), [1, 0])


def transposed_covariance_across_examples(inputs):
    """
    :param inputs: tensor with shape (example_count, stock_count, feature_count)
    :return: covariance.shape = (feature_count, stock_count, stock_count)
    """
    return tf.transpose(tfp.stats.covariance(inputs, inputs, sample_axis=0, event_axis=1), [2, 0, 1])


def transposed_variance_across_examples(inputs):
    """
    :param inputs: tensor with shape (example_count, stock_count, feature_count)
    :return: variance.shape = (feature_count, stock_count)
    """
    return tf.transpose(tfp.stats.variance(inputs, sample_axis=0), [1, 0])


def average_of_absolute_values(tensor):
    return tf.reduce_mean(tf.abs(tensor))


def average_kld(means, covs):
    return tf.reduce_mean(kl_divergence_for_multivariate_gaussian(*means, *covs))


def kl_divergence_for_multivariate_gaussian(mean0, mean1, cov0, cov1):
    """
    :param mean0: shape = (feature_count, stock_count)
    :param mean1: shape = (feature_count, stock_count)
    :param cov0: shape = (feature_count, stock_count, stock_count)
    :param cov1: shape = (feature_count, stock_count, stock_count)
    :return: shape = (feature_count)
    """
    dim = mean1.shape[1]
    mean_diff = mean1 - mean0
    inverse_cov1 = tf.linalg.inv(cov1)

    return (tf.linalg.trace(tf.matmul(inverse_cov1, cov0))
            + batch_dot_product(mean_diff, tf.linalg.matvec(inverse_cov1, mean_diff))
            - dim
            + tf.linalg.logdet(cov1) - tf.linalg.logdet(cov0)) / 2


# def load_saved_model(model: tf.keras.Model, file_path: str):
#     # need to build our models first
#     inputs_seq = tf.ones(shape=(BATCH_SIZE, WINDOW_LENGTH, STOCK_COUNT, INPUT_SIZE))
#     shared_input_seq = tf.ones(shape=(BATCH_SIZE, WINDOW_LENGTH, SHARED_INPUT_SIZE))
#     model(inputs_seq, shared_input_seq, BATCH_SIZE)
#
#     model.load_weights(file_path)
#     return model
#
#
# def mean_and_variance_from_sample(sample):
#     mean, variance = tf.nn.moments(x=sample, axes=[-1])
#     unbiased_variance_estimator = (BATCH_SIZE / (BATCH_SIZE - 1)) * variance
#     return mean, unbiased_variance_estimator


# TODO check (in colab) if mins and maxs are still broken in latest TF2
def reduce_min(tensor, axis=None):
    """Temporary fix while tf.reduce_min is broken"""
    min_of_non_nans = tf.reduce_min(tensor, axis)
    return tf.where(has_nan(tensor, axis), nans_like(min_of_non_nans), min_of_non_nans)


def reduce_max(tensor, axis=None):
    """Temporary fix while tf.reduce_max is broken"""
    max_of_non_nans = tf.reduce_max(tensor, axis)
    return tf.where(has_nan(tensor, axis), nans_like(max_of_non_nans), max_of_non_nans)


def minimum(t1, t2):
    """Temporary fix while tf.minimum is broken"""
    minimum_of_non_nans = tf.minimum(t1, t2)
    nan_locations = tf.logical_or(tf.math.is_nan(t1), tf.math.is_nan(t2))
    return tf.where(nan_locations, nans_like(minimum_of_non_nans), minimum_of_non_nans)


def maximum(t1, t2):
    """Temporary fix while tf.maximum is broken"""
    maximum_of_non_nans = tf.maximum(t1, t2)
    nan_locations = tf.logical_or(tf.math.is_nan(t1), tf.math.is_nan(t2))
    return tf.where(nan_locations, nans_like(maximum_of_non_nans), maximum_of_non_nans)


def has_nan(tensor, axis=None):
    return tf.reduce_any(tf.math.is_nan(tensor), axis)


def nans_like(tensor):
    return nans(tensor.shape, tensor.dtype)


def nans(shape, dtype):
    return tf.broadcast_to(tf.cast(NAN, dtype), shape)
