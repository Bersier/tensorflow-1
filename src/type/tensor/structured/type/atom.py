import abc
from enum import Enum, auto

from src.type.tensor.structured.type.core import Type


class Atom(abc.ABC, Type):
    pass


class Dynamic(Atom):
    pass


class Static(Enum, Atom):
    Absent = auto()
    Present = auto()
