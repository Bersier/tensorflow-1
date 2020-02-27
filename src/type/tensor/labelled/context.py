import abc

from src.type.tensor.labelled.axis import Axis


class Context(abc.ABC):
    @abc.abstractmethod
    def length(self, axis: Axis) -> int:
        pass
