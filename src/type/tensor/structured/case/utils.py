from src.commons.imports import tf
from src.commons.python.core import todo
from src.type.tensor.structured.case.core import View, Root
from src.type.tensor.structured.type.core import T


def new_root(tensor: tf.Tensor, of_type: T) -> Root:
    return todo(tensor, of_type)


def new_view(tensor: tf.Tensor, start_axis: int, view_type: T) -> View:
    return todo(tensor, start_axis, view_type)
