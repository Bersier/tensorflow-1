import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

from src.commons.imports import tf
from src.commons.python.record import NamedPair, NamedTriple


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
class Tensor(Generic[T], Type):
    type: T
    shape: Sequence[int]


@dataclass(frozen=True)
class Sum2(NamedPair[T1, T2], Type):
    pass


@dataclass(frozen=True)
class Sum3(NamedTriple[T1, T2, T3], Type):
    pass


@dataclass(frozen=True)
class Prd2(NamedPair[T1, T2], Type):
    pass


@dataclass(frozen=True)
class Prd3(NamedTriple[T1, T2, T3], Type):
    pass
