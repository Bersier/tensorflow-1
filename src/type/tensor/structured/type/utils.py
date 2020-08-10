import abc
from dataclasses import dataclass
from enum import Enum, auto

from typing import TypeVar, Sequence
from src.type.tensor.structured.type.core import Type


@dataclass(frozen=True)
class TypeClass(abc.ABC):
    pass


@dataclass(frozen=True)
class Plain(TypeClass):
    pass


@dataclass(frozen=True)
class Numeric(TypeClass):
    pass


@dataclass(frozen=True)
class Boolean(TypeClass):
    pass


C = TypeVar('C', covariant=True, bound=TypeClass)

N = TypeVar('N', covariant=True, bound=Type[Numeric])
N1 = TypeVar('N1', covariant=True, bound=Type[Numeric])
N2 = TypeVar('N2', covariant=True, bound=Type[Numeric])
N3 = TypeVar('N3', covariant=True, bound=Type[Numeric])
