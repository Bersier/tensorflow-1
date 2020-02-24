from src.commons.tensorflow.utils import TENSOR_FUNCTION

from src.commons.imports import tf
from src.commons.tensorflow.maker import bernoulli


def tile_tensor_at_zeroth_dimension(tensor: tf.Tensor, count: int) -> tf.Tensor:
    multiples = tf.squeeze(
        tf.one_hot(indices=[0], depth=len(tensor.shape), on_value=count, off_value=1, dtype=tf.int32))
    return tf.tile(tensor, multiples)


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


def with_noise(noise: tf.Tensor, noise_proportion: float) -> TENSOR_FUNCTION:
    assert tf.rank(noise) == 0

    def closure(t: tf.Tensor) -> tf.Tensor:
        mask = bernoulli(t.shape, noise_proportion)
        broadcast_noise = tf.broadcast_to(noise, t.shape)
        return tf.where(mask, broadcast_noise, t)

    return closure
