import abc
from dataclasses import dataclass
from typing import TypeVar

from src.type.tensor.structured.type import core
from src.type.tensor.structured.type.core import Type, Primitive, Tensor


@dataclass(frozen=True)
class Boolean(abc.ABC, Type):
    """This would be a type class if Python supported them."""
    pass


B = TypeVar('B', covariant=True, bound=Boolean)
B1 = TypeVar('B1', covariant=True, bound=Boolean)
B2 = TypeVar('B2', covariant=True, bound=Boolean)
B3 = TypeVar('B3', covariant=True, bound=Boolean)


@dataclass(frozen=True)
class Primitive(Primitive, Boolean):
    pass


@dataclass(frozen=True)
class Tensor(Tensor[B], Boolean):
    pass


@dataclass(frozen=True)
class Sum2(core.Sum2[B1, B2], Boolean):
    pass


@dataclass(frozen=True)
class Sum3(core.Sum3[B1, B2, B3], Boolean):
    pass


@dataclass(frozen=True)
class Prd2(core.Prd2[B1, B2], Boolean):
    pass


@dataclass(frozen=True)
class Prd3(core.Prd3[B1, B2, B3], Boolean):
    pass
