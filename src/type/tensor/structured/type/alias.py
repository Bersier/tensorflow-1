from src.commons.imports import tf
from src.commons.python import name
from src.type.tensor.structured.type import make, atom
from src.type.tensor.structured.type.core import Type


def option(of: Type) -> Type:
    return make.sum2((atom.Static.Present, of), atom.Static.Absent)


# noinspection PyArgumentList
def scalar(of: tf.dtypes.DType) -> Type:
    return make.primitive(of)


def vector(of_type: Type, length: int) -> Type:
    return of_type.tensor(shape=[length]).tagged(tag=name.Static.Vector)


def matrix(of_type: Type, row_count: int, column_count: int) -> Type:
    return of_type.tensor(shape=[row_count, column_count]).tagged(tag=name.Static.Matrix)
