from typing import Mapping, Tuple, List

from src.commons.imports import tf
from src.commons.python.core import fill
from src.commons.tensorflow.typealias import TENSOR_PAIR


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
