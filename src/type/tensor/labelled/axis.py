import abc
from enum import Enum, auto


class Axis(abc.ABC):
    pass


class Dynamic(abc.ABC, Axis):
    def __init__(self):
        pass


class Static(Enum, Axis):
    Feature = auto()
    Batch = auto()
    Run = auto()
    Example = auto()
    State = auto()
    Output = auto()
