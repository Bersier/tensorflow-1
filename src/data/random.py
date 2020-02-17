from typing import Tuple

import numpy as np

from src.data.sizeddataset import SizedDataset
from src.data.utils import get_dataset_from_numpy

FEATURE_COUNT = 32
CLASS_COUNT = 20


def random_dataset(size: int) -> Tuple[SizedDataset, SizedDataset]:
    data = np.random.random((size, FEATURE_COUNT))
    labels = random_one_hot_labels((size, CLASS_COUNT))
    return get_dataset_from_numpy((data, labels))


def random_one_hot_labels(shape: Tuple[int, int]) -> np.ndarray:
    batch_size, class_count = shape
    classes = np.random.randint(0, class_count, batch_size)
    labels = np.zeros((batch_size, class_count))
    labels[np.arange(batch_size), classes] = 1
    return labels