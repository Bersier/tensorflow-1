import abc
from dataclasses import dataclass
from typing import Generic

from src.commons.python.zipper import T


@dataclass(frozen=True)
class List(abc.ABC, Generic[T]):
    pass


@dataclass(frozen=True)
class Empty(List):
    pass


@dataclass(frozen=True)
class Cons(List[T]):
    head: T
    tail: List[T]
