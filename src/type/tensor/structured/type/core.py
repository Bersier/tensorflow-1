from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

from src.commons.imports import tf
from src.commons.python.name import Name
from src.commons.python.record import NamedPair, NamedTriple
from src.type.tensor.structured.type.utils import TypeClass, C, Plain


@dataclass(frozen=True)
class Type(abc.ABC, Generic[C]):

    def tensor(self, shape: Sequence[int]) -> Tensor:
        return Tensor(self, shape)


T = TypeVar('T', covariant=True, bound=Type)
T1 = TypeVar('T1', covariant=True, bound=Type)
T2 = TypeVar('T2', covariant=True, bound=Type)
T3 = TypeVar('T3', covariant=True, bound=Type)

SELF_T = TypeVar('SELF_T', covariant=True, bound=Type)


@dataclass(frozen=True)
class Primitive(abc.ABC, Type[C]):
    dtype: tf.dtypes.DType


@dataclass(frozen=True)
class Tensor(Type[C], Generic[T, C]):
    type: T
    shape: Sequence[int]


@dataclass(frozen=True)
class Sum2(NamedPair[T1, T2], Type[Plain]):
    pass


@dataclass(frozen=True)
class Sum3(NamedTriple[T1, T2, T3], Type[Plain]):
    pass


@dataclass(frozen=True)
class Prd2(NamedPair[T1, T2], Type[Plain]):
    pass


@dataclass(frozen=True)
class Prd3(NamedTriple[T1, T2, T3], Type[Plain]):
    pass
