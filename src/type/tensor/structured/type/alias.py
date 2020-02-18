from src.commons.imports import tf
from src.type.tensor.structured.type import atom, name
from src.type.tensor.structured.type.core import Type, Sum, Tensor, Tagged, Primitive


def option(of: Type) -> Type:
    return Sum(union=[of, atom.Static.Absent])


def scalar(of: tf.dtypes.DType) -> Type:
    return Primitive(of)


def vector(of: Type, length: int) -> Type:
    return Tagged(Tensor(type=of, shape=[length]), name=name.Static.Vector)


def matrix(of: Type, row_count: int, column_count: int) -> Type:
    return Tagged(Tensor(type=of, shape=[row_count, column_count]), name=name.Static.Matrix)
