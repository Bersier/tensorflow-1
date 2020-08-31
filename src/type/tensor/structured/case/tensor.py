from __future__ import annotations

from src.commons.imports import tf
from src.commons.tensorflow.getter import slice_along
from src.type.tensor.structured.case import core
from src.type.tensor.structured.case.utils import new_view
from src.type.tensor.structured.type.core import Tensor


class View(core.View[Tensor]):

    def __getitem__(self, *args):
        """
        Creates a View of a new structure
        """
        axes_to_squeeze = []
        ranges = {}
        i = self._start_axis
        for arg in args:
            if type(arg) == int:
                axes_to_squeeze.append(i)
                ranges[i] = (arg, arg + 1)
            elif type(arg) == slice:
                assert arg.step == 1
                ranges[i] = (arg.start, arg.stop)
            else:
                raise TypeError
            i += 1

        tensor = slice_along(self._tensor, ranges)
        tensor = tf.squeeze(tensor, axis=axes_to_squeeze)
        mask = self._mask
        if mask:
            mask = slice_along(mask, ranges)
            mask = tf.squeeze(mask, axis=axes_to_squeeze)
        view_type = self._view_type.with_updates(shape=tf.shape(tensor)[self._start_axis:])
        return View(tensor, self._start_axis, mask, view_type)

    def element_view(self):
        """
        Changes the focus without changing the structure
        """
        return new_view(
            tensor=self._tensor,
            start_axis=self._start_axis + len(self._view_type.focus.shape),
            view_type=self._view_type.type
        )
