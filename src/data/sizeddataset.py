from dataclasses import dataclass

from src.imports import tf

Dataset = tf.data.Dataset


@dataclass(frozen=True)
class SizedDataset:
    data: Dataset
    size: int
