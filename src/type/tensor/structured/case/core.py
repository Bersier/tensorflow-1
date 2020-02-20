from __future__ import annotations

import abc
from typing import Generic

from src.commons.imports import tf
from src.commons.python import todo
from src.type.tensor.structured.case.utils import ViewT, RootT, new_root
from src.type.tensor.structured.type.core import T


def is_valid_type(tensor: tf.Tensor, start_axis: int, of_type: T) -> bool:
    return todo(tensor, start_axis, of_type)


class View(abc.ABC, Generic[RootT, ViewT]):
    def __init__(self, tensor: tf.Tensor, start_axis: int, root_type: RootT, view_type: ViewT):
        assert is_valid_type(tensor, 0, root_type)
        assert is_valid_type(tensor, start_axis, view_type)
        self._tensor = tensor
        self._start_axis = start_axis
        self._root_type = root_type
        self._view_type = view_type

    def root(self) -> Root:
        return new_root(self._tensor, self._root_type)


class Root(abc.ABC, View[T, T]):
    def __init__(self, tensor: tf.Tensor, of_type: RootT):
        assert is_valid_type(tensor, 0, of_type)
        self._tensor = tensor
        self._start_axis = 0
        self._root_type = of_type
        self._view_type = of_type

    def root(self) -> Root:
        return self

