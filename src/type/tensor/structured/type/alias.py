from src.commons.imports import tf
from src.type.tensor.structured.type import name
from src.type.tensor.structured.type.core import Type, Tensor, Tagged, Numeric


# def option(of: Type) -> Type:
#     return Sum2(of, atom.Static.Absent)


# noinspection PyArgumentList
def scalar(of: tf.dtypes.DType) -> Type:
    return Numeric(of)


def vector(of: Type, length: int) -> Type:
    return Tagged(Tensor(type=of, shape=[length]), name=name.Static.Vector)


def matrix(of: Type, row_count: int, column_count: int) -> Type:
    return Tagged(Tensor(type=of, shape=[row_count, column_count]), name=name.Static.Matrix)
