import abc

from src.types.labelledtensor.axes import Axis


class Context(abc.ABC):
    @abc.abstractmethod
    def length(self, axe: Axis) -> int:
        pass
