import abc
from enum import Enum, auto

from src.type.tensor.structured.type.core import Type


class Atom(abc.ABC, Type):
    pass


class Dynamic(abc.ABC, Atom):
    def __init__(self):
        pass


class Static(Enum, Atom):
    Absent = auto()
