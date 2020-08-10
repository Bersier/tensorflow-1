from __future__ import annotations

import abc
from typing import Generic, Optional, TypeVar, Any

import tensorflow as tf

from src.commons.python.core import todo
from src.commons.python.list import List
from src.commons.python.zipper import Zipper
from src.type.tensor.structured.case.utils import Root
from src.type.tensor.structured.type.core import T, Type

IS_ROOT = TypeVar('IS_ROOT', covariant=True, bound=Root)


def is_valid_type(tensor: tf.Tensor, start_axis: int, of_type: T) -> bool:
    return todo(tensor, start_axis, of_type)


class View(abc.ABC, Generic[T, IS_ROOT]):
    def __init__(self, tensor: tf.Tensor, start_axis: int, mask: Optional[tf.Tensor], view_type: Zipper[T]):
        assert is_valid_type(tensor, start_axis, view_type.focus)
        self._tensor = tensor
        self._start_axis = start_axis
        self._view_type = view_type
        self._mask = mask

    @classmethod
    def _root(cls, t: tf.Tensor, of_type: Type) -> View[Any, Root.Yes]:
        return cls(t, 0, None, Zipper(path=List.empty(), focus=of_type))

    def root(self) -> View[Any, Root.Yes]:
        return self._root(self._tensor, self._view_type.root())

    def type(self):
        return self._view_type.focus
