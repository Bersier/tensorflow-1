from __future__ import annotations

from src.commons.imports import tf
from src.commons.python import todo
from src.type.tensor.structured.type.core import Type, Tensor


class StructuredTensor:
    """
    Normal Tensors are flat; they don't have depth. So nested structures cannot be represented explicitly.
    This lack of recursion leads to a lack of compositionality.
    An operation cannot be naturally applied to nested sub-tensors.

    This class is an attempt at fixing this, by keeping track of the structure separately,
    while internally representing everything as flat tensors.
    """

    def __init__(self, tensor: tf.Tensor, of_type: Type):
        self._tensor = tensor
        self._type = of_type


class TensorTensor:

    def __init__(self, tensor: tf.Tensor, of_type: Tensor):
        self._tensor = tensor
        self._type = of_type

    def map(self, function) -> TensorTensor:
        return todo(self, function)
