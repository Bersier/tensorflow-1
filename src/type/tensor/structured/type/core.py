import abc
from dataclasses import dataclass
from typing import List

from src.commons.imports import tf
from src.type.tensor.structured.type.name import Name


@dataclass(frozen=True)
class Type(abc.ABC):
    pass


@dataclass(frozen=True)
class Primitive(Type):
    dtype: tf.dtypes.DType


@dataclass(frozen=True)
class Tensor(Type):
    type: Type
    shape: List[int]


@dataclass(frozen=True)
class Sum(Type):
    union: List[Type]


@dataclass(frozen=True)
class Product(Type):
    product: List[Type]


@dataclass(frozen=True)
class Tagged(Type):
    type: Type
    name: Name
