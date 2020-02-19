import abc
from typing import Generic

from src.commons.imports import tf
from src.type.tensor.structured.type.core import T


class Structure(abc.ABC, Generic[T]):

    def __init__(self, tensor: tf.Tensor, of_type: T):
        self._tensor = tensor
        self._type = of_type
