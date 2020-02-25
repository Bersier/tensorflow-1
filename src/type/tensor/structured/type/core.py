import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

from src.commons.imports import tf
from src.type.tensor.structured.type.name import Name


@dataclass(frozen=True)
class Type(abc.ABC):
    pass


T = TypeVar('T', covariant=True, bound=Type)
T1 = TypeVar('T1', covariant=True, bound=Type)
T2 = TypeVar('T2', covariant=True, bound=Type)
T3 = TypeVar('T3', covariant=True, bound=Type)


@dataclass(frozen=True)
class Primitive(abc.ABC, Type):
    dtype: tf.dtypes.DType


@dataclass(frozen=True)
class Numeric(Primitive):
    pass


@dataclass(frozen=True)
class Boolean(Primitive):
    pass


@dataclass(frozen=True)
class Tensor(Generic[T], Type):
    type: T
    shape: Sequence[int]


@dataclass(frozen=True)
class Tagged(Generic[T], Type):
    type: T
    name: Name


# TODO Implement Sum and Product.
#   Sum and product implementation requires more bookkeeping,
#   general masks for sums, and bound/range masks for products.
# @dataclass(frozen=True)
# class Sum2(Generic[T1, T2], Type):
#     type1: T1
#     type2: T2
#
#
# @dataclass(frozen=True)
# class Sum3(Generic[T1, T2, T3], Type):
#     type1: T1
#     type2: T2
#     type3: T3
#
#
# @dataclass(frozen=True)
# class Prd2(Generic[T1, T2], Type):
#     type1: T1
#     type2: T2
#
#
# @dataclass(frozen=True)
# class Prd3(Generic[T1, T2, T3], Type):
#     type1: T1
#     type2: T2
#     type3: T3
