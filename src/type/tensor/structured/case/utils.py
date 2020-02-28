from enum import Enum, auto

from src.commons.imports import tf
from src.commons.python.core import todo
from src.commons.python.zipper import Zipper
from src.type.tensor.structured.case.core import View
from src.type.tensor.structured.type.core import T


def new_view(tensor: tf.Tensor, start_axis: int, view_type: Zipper[T]) -> View:
    return todo(tensor, start_axis, view_type)


class Root(Enum):
    No = auto(),
    Yes = auto(),
