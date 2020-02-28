from __future__ import annotations

from src.commons.imports import tf
from src.type.tensor.structured.case import core
from src.type.tensor.structured.case.core import Root
from src.type.tensor.structured.type.numeric import N


class View(core.View[N]):

    # noinspection PyProtectedMember
    def __add__(self, other: Root[N]):
        assert self.type() == other.type()
        sum_tensor = self._tensor + other._tensor
        if self._mask:
            sum_tensor = tf.where(self._mask, sum_tensor, self._tensor)
        return View(sum_tensor, self._start_axis, self._mask, self._view_type)