from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, Any

from src.commons.python.zipper import T


@dataclass(frozen=True)
class List(abc.ABC, Generic[T]):

    @staticmethod
    def empty() -> List[Any]:
        # noinspection PyTypeChecker
        return Empty


@dataclass(frozen=True)
class Empty(List[Any]):
    pass


@dataclass(frozen=True)
class Cons(List[T]):
    head: T
    tail: List[T]
