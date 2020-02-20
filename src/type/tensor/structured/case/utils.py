from typing import TypeVar

from src.commons.imports import tf
from src.commons.python import todo
from src.type.tensor.structured.case.core import View, Root
from src.type.tensor.structured.type.core import Type, T

ViewT = TypeVar('ViewT', covariant=True, bound=Type)
RootT = TypeVar('RootT', covariant=True, bound=Type)


def new_root(tensor: tf.Tensor, of_type: T) -> Root:
    return todo(tensor, of_type)


def new_view(tensor: tf.Tensor, start_axis: int, root_type: RootT, view_type: ViewT) -> View:
    return todo(tensor, start_axis, root_type, view_type)
