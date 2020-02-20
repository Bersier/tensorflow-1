from __future__ import annotations

import abc
from typing import Generic, TypeVar

from src.commons.imports import tf
from src.commons.python import todo
from src.type.tensor.structured.type.core import T, Type

PointeeT = TypeVar('PointeeT', covariant=True, bound=Type)
RootT = TypeVar('RootT', covariant=True, bound=Type)


def is_valid_type(tensor: tf.Tensor, start_axis: int, of_type: T) -> bool:
    return todo(tensor, start_axis, of_type)


class Structure(abc.ABC, Generic[T]):

    def __init__(self, tensor: tf.Tensor, start_axis: int, of_type: T):
        assert is_valid_type(tensor, start_axis, of_type)
        self._tensor = tensor
        self._start_axis = start_axis
        self._type = of_type

    def root(self) -> Structure:
        pass


class Pointer(abc.ABC, Generic[RootT, PointeeT]):
    def __init__(self, tensor: tf.Tensor, start_axis: int, root_type: RootT, pointee_type: PointeeT):
        assert is_valid_type(tensor, start_axis, of_type)
        self._tensor = tensor
        self._start_axis = start_axis
        self._type = of_type

    def root(self) -> Structure:
        pass
