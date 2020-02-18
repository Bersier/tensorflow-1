import abc
from enum import Enum, auto


class Name(abc.ABC):
    pass


class Dynamic(abc.ABC, Name):
    def __init__(self):
        pass


class Static(Enum, Name):
    Guessed = auto()
    Matrix = auto()
    Missing = auto()
    Vector = auto()
