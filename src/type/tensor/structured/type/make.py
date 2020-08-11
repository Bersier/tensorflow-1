from typing import Tuple

from src.commons.imports import tf
from src.commons.python.name import Name
from src.type.tensor.structured.type.core import Primitive, T1, T2
from src.type.tensor.structured.type.utils import Numeric

NUMERIC_DTYPES = {
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
    tf.dtypes.int8,
    tf.dtypes.int16,
    tf.dtypes.int32,
    tf.dtypes.int64,
    # TODO
}


# # noinspection PyArgumentList
# def sum2(first_or_second: Tuple[Name, T1], second_or_first: Tuple[Name, T2]):
#     if isinstance(first_or_second[1], Numeric) and isinstance(second_or_first[1], Numeric):
#         return numeric.Sum2(first_or_second, second_or_first)
#     if isinstance(first_or_second[1], Boolean) and isinstance(second_or_first[1], Boolean):
#         return boolean.Sum2(first_or_second, second_or_first)
#     return core.Sum2(first_or_second, second_or_first)
#
#
# # noinspection PyArgumentList
# def primitive(of: tf.dtypes.DType) -> Primitive:
#     if of in NUMERIC_DTYPES:
#         return numeric.Primitive(of)
#     if of == tf.dtypes.bool:
#         return boolean.Primitive(of)
#     return core.Primitive(of)
