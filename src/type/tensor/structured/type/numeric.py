from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeVar, Sequence

from src.commons.python.name import Name
from src.type.tensor.structured.type import core
from src.type.tensor.structured.type.core import Type


@dataclass(frozen=True)
class Numeric(abc.ABC, Type):
    """This would be a type class if Python supported them."""

    # noinspection PyArgumentList
    def tensor(self, shape: Sequence[int]) -> Tensor:
        return Tensor(self, shape)

    # noinspection PyArgumentList
    def tagged(self, tag: Name) -> Tagged:
        return Tagged(tag, self)


N = TypeVar('N', covariant=True, bound=Numeric)
N1 = TypeVar('N1', covariant=True, bound=Numeric)
N2 = TypeVar('N2', covariant=True, bound=Numeric)
N3 = TypeVar('N3', covariant=True, bound=Numeric)


@dataclass(frozen=True)
class Primitive(core.Primitive, Numeric):
    pass


@dataclass(frozen=True)
class Tensor(core.Tensor[N], Numeric):
    pass


@dataclass(frozen=True)
class Tagged(core.Tagged[N], Numeric):
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
