from dataclasses import dataclass

from numpy import ndarray


@dataclass(frozen=True)
class Gaussian:
    mean: ndarray
    cov: ndarray
