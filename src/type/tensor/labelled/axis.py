import abc
from enum import Enum, auto


class Axis(abc.ABC):
    pass


class Dynamic(Axis):
    pass


class Static(Enum, Axis):
    Feature = auto()
    Batch = auto()
    Run = auto()
    Example = auto()
    State = auto()
    Output = auto()
