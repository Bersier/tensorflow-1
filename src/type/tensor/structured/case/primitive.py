from __future__ import annotations

import abc
from typing import TypeVar

from src.type.tensor.structured.case import core
from src.type.tensor.structured.type.core import Primitive

T = TypeVar('T', covariant=True, bound=Primitive)


class Structure(abc.ABC, core.Structure[T]):
    pass
