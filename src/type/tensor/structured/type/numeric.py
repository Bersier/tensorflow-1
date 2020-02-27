import abc
from dataclasses import dataclass
from typing import TypeVar

from src.type.tensor.structured.type import core
from src.type.tensor.structured.type.core import Type, Primitive, Tensor


@dataclass(frozen=True)
class Numeric(abc.ABC, Type):
    """This would be a type class if Python supported them."""
    pass


N = TypeVar('N', covariant=True, bound=Numeric)
N1 = TypeVar('N1', covariant=True, bound=Numeric)
N2 = TypeVar('N2', covariant=True, bound=Numeric)
N3 = TypeVar('N3', covariant=True, bound=Numeric)


@dataclass(frozen=True)
class Primitive(Primitive, Numeric):
    pass


@dataclass(frozen=True)
class Tensor(Tensor[N], Numeric):
    pass


@dataclass(frozen=True)
class Sum2(core.Sum2[N1, N2], Numeric):
    pass


@dataclass(frozen=True)
class Sum3(core.Sum3[N1, N2, N3], Numeric):
    pass


@dataclass(frozen=True)
class Prd2(core.Prd2[N1, N2], Numeric):
    pass


@dataclass(frozen=True)
class Prd3(core.Prd3[N1, N2, N3], Numeric):
    pass
