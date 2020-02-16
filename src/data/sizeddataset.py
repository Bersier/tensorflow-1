from dataclasses import dataclass
from typing import Tuple

from numpy import ndarray
from tensorflow.data import Dataset


@dataclass(frozen=True)
class SizedDataset:
    data: Dataset
    size: int


def from_numpy(numpy_set: Tuple[ndarray, ndarray]) -> SizedDataset:
    return SizedDataset(Dataset.from_tensor_slices(numpy_set), numpy_set[0][0])
