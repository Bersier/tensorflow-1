from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy import ma

from src.commons.python import product
from src.data.utils import from_numpy
from src.type.core import SizedDataset


@dataclass(frozen=True)
class DatasetSpec:
    size: int
    feature_count: int
    class_count: int
    nan_proportion: float = 0.0


def random_dataset(spec: DatasetSpec) -> SizedDataset:
    features = random_features_with_nans(
        shape=[spec.size, spec.feature_count],
        nan_proportion=spec.nan_proportion
    )
    labels = np.random.randint(
        size=[spec.size, 1],
        low=0,
        high=spec.class_count
    )
    return from_numpy((features, labels))


def random_features_with_nans(shape, nan_proportion):
    features = np.random.random(shape)
    nan_mask = random_mask(shape, nan_proportion)
    masked_data = ma.array(features, mask=nan_mask)
    return masked_data.filled(np.NAN)


def random_mask(shape: List[int], true_proportion) -> np.ndarray:
    count = product(shape)
    false_count = int(count * (1 - true_proportion))
    true_count = count - false_count
    mask = np.concatenate([
        np.ones(true_count, dtype=bool),
        np.zeros(false_count, dtype=bool)
    ])
    np.random.shuffle(mask)
    return np.reshape(mask, shape)


def random_one_hot_labels(shape: Tuple[int, int]) -> np.ndarray:
    batch_size, class_count = shape
    classes = np.random.randint(0, class_count, batch_size)
    labels = np.zeros((batch_size, class_count))
    labels[np.arange(batch_size), classes] = 1
    return labels
