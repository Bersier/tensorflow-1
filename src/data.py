from typing import Tuple

import numpy as np
from tensorflow.data import Dataset

TRAINING_SET_SIZE = 2000
VALIDATION_SET_SIZE = 100
FEATURE_COUNT = 32
CLASS_COUNT = 20

BATCH_SIZE = 128


def get_datasets() -> Tuple[Dataset, Dataset]:
    dataset = random_dataset(TRAINING_SET_SIZE).shuffle(
        buffer_size=TRAINING_SET_SIZE,
        reshuffle_each_iteration=True
    ).batch(BATCH_SIZE)

    val_dataset = random_dataset(VALIDATION_SET_SIZE).batch(BATCH_SIZE)
    return dataset, val_dataset


def random_dataset(size: int) -> Dataset:
    data = np.random.random((size, FEATURE_COUNT))
    labels = random_one_hot_labels((size, CLASS_COUNT))
    return Dataset.from_tensor_slices((data, labels))


def random_one_hot_labels(shape: Tuple[int, int]) -> np.ndarray:
    batch_size, class_count = shape
    classes = np.random.randint(0, class_count, batch_size)
    labels = np.zeros((batch_size, class_count))
    labels[np.arange(batch_size), classes] = 1
    return labels
