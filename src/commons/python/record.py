from dataclasses import dataclass
from typing import Generic, Tuple, cast, TypeVar

from src.commons.python.name import Name

T1 = TypeVar('T1', covariant=True)
T2 = TypeVar('T2', covariant=True)
T3 = TypeVar('T3', covariant=True)


@dataclass(frozen=True)
class NamedPair(Generic[T1, T2]):
    _first: Tuple[Name, T1]
    _second: Tuple[Name, T2]

    def __init__(self, first_or_second: Tuple[Name, T1], second_or_first: Tuple[Name, T2]):
        id1 = id(first_or_second[0])
        id2 = id(second_or_first[0])
        if id1 > id2:
            first_or_second, second_or_first = (second_or_first, first_or_second)
        object.__setattr__(self, 'first', first_or_second)
        object.__setattr__(self, 'second', second_or_first)

    def __getitem__(self, arg):
        if type(arg) is Name:
            name = cast(Name, arg)
            if name == self._first[0]:
                return self._first[1]
            if name == self._second[0]:
                return self._second[1]
            raise ValueError
        raise TypeError


@dataclass(frozen=True)
class NamedTriple(Generic[T1, T2, T3]):
    _first: Tuple[Name, T1]
    _second: Tuple[Name, T2]
    _third: Tuple[Name, T3]

    def __init__(self, named1: Tuple[Name, T1], named2: Tuple[Name, T2], named3: Tuple[Name, T3]):
        id1 = id(named1[0])
        id2 = id(named2[0])
        id3 = id(named3[0])
        if id1 > id2:
            named1, named2 = (named2, named1)
        if id2 > id3:
            named2, named3 = (named3, named2)
        if id1 > id2:
            named1, named2 = (named2, named1)
        object.__setattr__(self, 'first', named1)
        object.__setattr__(self, 'second', named2)
        object.__setattr__(self, 'third', named3)

    def __getitem__(self, arg):
        if type(arg) == Name:
            name = cast(Name, arg)
            if name == self._first[0]:
                return self._first[1]
            if name == self._second[0]:
                return self._second[1]
            if name == self._third[0]:
                return self._third[1]
            raise ValueError
        raise TypeError
