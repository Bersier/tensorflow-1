from __future__ import annotations

from src.commons.imports import tf
from src.type.tensor.structured.case import core
from src.type.tensor.structured.case.core import IS_ROOT
from src.type.tensor.structured.case.utils import Root
from src.type.tensor.structured.type.numeric import N


class View(core.View[N, IS_ROOT]):

    # noinspection PyProtectedMember
    def __add__(self, other: View[N, Root.Yes]):
        assert self.type() == other.type()
        sum_tensor = self._tensor + other._tensor
        if self._mask:
            sum_tensor = tf.where(self._mask, sum_tensor, self._tensor)
        return View(sum_tensor, self._start_axis, self._mask, self._view_type)
