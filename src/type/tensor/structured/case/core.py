from __future__ import annotations

import abc
from typing import Generic, Optional

from src.commons.imports import tf
from src.commons.python.core import todo
from src.commons.python.zipper import Zipper, zipper_of
from src.type.tensor.structured.case.utils import new_root
from src.type.tensor.structured.type.core import T


def is_valid_type(tensor: tf.Tensor, start_axis: int, of_type: T) -> bool:
    return todo(tensor, start_axis, of_type)


class View(abc.ABC, Generic[T]):
    def __init__(self, tensor: tf.Tensor, start_axis: int, mask: Optional[tf.Tensor], view_type: Zipper[T]):
        assert is_valid_type(tensor, start_axis, view_type.focus)
        self._tensor = tensor
        self._start_axis = start_axis
        self._view_type = view_type
        self._mask = mask

    def root(self) -> Root:
        return new_root(self._tensor, self._view_type.root())


class Root(abc.ABC, View[T]):
    def __init__(self, tensor: tf.Tensor, of_type: T):
        assert is_valid_type(tensor, 0, of_type)
        self._tensor = tensor
        self._start_axis = 0
        self._mask = None
        self._view_type = zipper_of(of_type)

    def root(self) -> Root:
        return self

    def type(self):
        return self._view_type.focus
