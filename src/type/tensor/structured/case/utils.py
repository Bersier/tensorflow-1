from src.commons.imports import tf
from src.commons.python import todo
from src.type.tensor.structured.case.core import Structure
from src.type.tensor.structured.type.core import Type


def new_structure(tensor: tf.Tensor, start_axis: int, of_type: Type) -> Structure:
    return todo(tensor, start_axis, of_type)
