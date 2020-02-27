import abc
from enum import Enum, auto


class Name(abc.ABC):
    pass


class Dynamic(Name):
    pass


class Static(Enum, Name):
    Guessed = auto()
    Matrix = auto()
    Missing = auto()
    Vector = auto()
