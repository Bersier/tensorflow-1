from __future__ import annotations

from src.type.tensor.structured.case import numericimpl, tensor
from src.type.tensor.structured.type import numeric


class Tensor(tensor.View, numericimpl.View[numeric.Tensor]):
    pass
