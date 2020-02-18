import abc

from src.type.tensor.labelled.axes import Axis


class Context(abc.ABC):
    @abc.abstractmethod
    def length(self, axe: Axis) -> int:
        pass
