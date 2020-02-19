from __future__ import annotations

from src.type.tensor.structured.case import core
from src.type.tensor.structured.type.core import Tensor


class Structure(core.Structure[Tensor]):

    # noinspection PyProtectedMember
    def __add__(self, other: Structure):
        assert self._type == other._type
        return Structure(self._tensor + other._tensor, self._type)
