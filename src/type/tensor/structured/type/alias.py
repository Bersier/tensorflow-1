from src.commons.imports import tf
from src.commons.python import name
from src.type.tensor.structured.type import make, atom
from src.type.tensor.structured.type.core import Type


def option(of: Type) -> Type:
    return make.sum2((atom.Static.Present, of), atom.Static.Absent)


# noinspection PyArgumentList
def scalar(of: tf.dtypes.DType) -> Type:
    return make.primitive(of)
