from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeVar, Sequence

from src.commons.python.name import Name
from src.type.tensor.structured.type import core
from src.type.tensor.structured.type.core import Type


@dataclass(frozen=True)
class Boolean(abc.ABC, Type):
    """This would be a type class if Python supported them."""

    # noinspection PyArgumentList
    def tensor(self, shape: Sequence[int]) -> Tensor:
        return Tensor(self, shape)

    # noinspection PyArgumentList
    def tagged(self, tag: Name) -> Tagged:
        return Tagged(tag, self)


B = TypeVar('B', covariant=True, bound=Boolean)
B1 = TypeVar('B1', covariant=True, bound=Boolean)
B2 = TypeVar('B2', covariant=True, bound=Boolean)
B3 = TypeVar('B3', covariant=True, bound=Boolean)


@dataclass(frozen=True)
class Primitive(core.Primitive, Boolean):
    pass


@dataclass(frozen=True)
class Tensor(core.Tensor[B], Boolean):
    pass


@dataclass(frozen=True)
class Tagged(core.Tagged[B], Boolean):
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
