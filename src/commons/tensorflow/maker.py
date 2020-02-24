from typing import List

import tensorflow_probability as tfp

from src.commons.imports import tf

NAN = tf.math.log(-1.0)


def nans(shape: List[int], dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.broadcast_to(tf.cast(NAN, dtype), shape)


def nans_like(tensor: tf.Tensor) -> tf.Tensor:
    return nans(tensor.shape, tensor.dtype)


def bernoulli(shape: List[int], truth_probability: float = 0.5) -> tf.Tensor:
    distribution = tfp.distributions.Bernoulli(probs=truth_probability)
    return tf.cast(distribution.sample(shape), dtype=tf.dtypes.bool)
