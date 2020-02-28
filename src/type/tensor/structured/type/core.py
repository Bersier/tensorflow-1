from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

from src.commons.imports import tf
from src.type.tensor.structured.case import tensor
from src.type.tensor.structured.case.utils import Root
from src.commons.python.name import Name
from src.commons.python.record import NamedPair, NamedTriple
from src.type.tensor.structured.case.core import View


@dataclass(frozen=True)
class Type(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def _corresponding_view() -> type(View):
        pass

    def new_root(self: SELF_T, t: tf.Tensor) -> View[SELF_T, Root.Yes]:
        # noinspection PyProtectedMember
        return self._corresponding_view()._root(t, self)

    def tensor(self, shape: Sequence[int]) -> Tensor:
        return Tensor(self, shape)

    def tagged(self, tag: Name) -> Tagged:
        return Tagged(tag, self)


T = TypeVar('T', covariant=True, bound=Type)
T1 = TypeVar('T1', covariant=True, bound=Type)
T2 = TypeVar('T2', covariant=True, bound=Type)
T3 = TypeVar('T3', covariant=True, bound=Type)

SELF_T = TypeVar('SELF_T', covariant=True, bound=Type)


@dataclass(frozen=True)
class Primitive(abc.ABC, Type):
    dtype: tf.dtypes.DType


@dataclass(frozen=True)
class Tensor(Generic[T], Type):
    type: T
    shape: Sequence[int]

    @staticmethod
    def _corresponding_view() -> type(tensor.View):
        return tensor.View


@dataclass(frozen=True)
class Tagged(Generic[T], Type):
    tag: Name
    type: T


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
