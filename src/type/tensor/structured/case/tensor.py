from __future__ import annotations

from dataclasses import replace

from src.commons.imports import tf
from src.commons.tensorflow import slice_along
from src.type.tensor.structured.case import core
from src.type.tensor.structured.case.utils import new_structure
from src.type.tensor.structured.type.core import Tensor


class Structure(core.Structure[Tensor]):

    def __init__(self, tensor: tf.Tensor, axis_range, axis_map, of_type: Tensor):
        """
        :param tensor:
        :param axis_range:
        :param axis_map: maps logical axes to representation axes
        :param of_type:
        """
        super().__init__(tensor, axis_range.start, of_type)
        self._axis_range = axis_range
        self._axis_map = axis_map

    # noinspection PyProtectedMember
    def __add__(self, other: Structure):
        assert self._type == other._type
        return Structure(self._tensor + other._tensor, self._start_axis, self._type)

    def __getitem__(self, *args):
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
        of_type = replace(self._type, shape=tf.shape(tensor))
        return Structure(tensor, self._start_axis, of_type) # TODO: needs new axis mapping!

    def element(self):
        return new_structure(self._tensor, self._start_axis + )