from __future__ import annotations

from src.type.tensor.structured.case import primitive
from src.type.tensor.structured.type.core import Numeric


# TODO should be a type class, ideally, so that it can be applied to e.g. Tensor...
# Have NumericTensor and BooleanTensor,
# where both NumericTensor and "PrimitiveNumeric" would inherit from "NumericI"?
class Structure(primitive.Structure[Numeric]):

    # noinspection PyProtectedMember
    def __add__(self, other: Structure):
        assert self._type == other._type
        return Structure(self._tensor + other._tensor, self._type)
