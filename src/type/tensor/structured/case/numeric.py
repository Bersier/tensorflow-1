from __future__ import annotations

from src.type.tensor.structured.type import core
from src.type.tensor.structured.case import numericimpl, tensor
from src.type.tensor.structured.type.utils import Numeric


class Tensor(tensor.View, numericimpl.View[core.Tensor[core.T, Numeric]]):
    pass
