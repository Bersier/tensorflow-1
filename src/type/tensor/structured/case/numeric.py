from __future__ import annotations

from src.type.tensor.structured.case import primitive
from src.type.tensor.structured.case.core import Root
from src.type.tensor.structured.type.core import Numeric


# TODO should be a type class, ideally, so that it can be applied to e.g. Tensor...
#   Have NumericTensor and BooleanTensor,
#   where both NumericTensor and "PrimitiveNumeric" would inherit from "NumericI"?
class View(primitive.View[Numeric]):

    # noinspection PyProtectedMember
    def __add__(self, other: Root[Numeric]):
        assert self._view_type.focus == other.type()
        return View(self._tensor + other._tensor, self._start_axis, self._view_type)
