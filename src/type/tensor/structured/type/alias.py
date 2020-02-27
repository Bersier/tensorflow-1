import src.commons.python.name
from src.commons.imports import tf
from src.type.tensor.structured.type.core import Type, Tensor
from src.type.tensor.structured.type.numeric import Numeric


# def option(of: Type) -> Type:
#     return Sum2(of, atom.Static.Absent)


# noinspection PyArgumentList
def scalar(of: tf.dtypes.DType) -> Type:
    return Numeric(of)


def vector(of: Type, length: int) -> Type:
    return Tagged(Tensor(type=of, shape=[length]), name=src.commons.python.name.Static.Vector)


def matrix(of: Type, row_count: int, column_count: int) -> Type:
    return Tagged(Tensor(type=of, shape=[row_count, column_count]), name=src.commons.python.name.Static.Matrix)
