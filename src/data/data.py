from typing import Tuple

import numpy as np
from tensorflow.data import Dataset

from src.data.sizeddataset import from_numpy
from src.data.utils import split_dataset
from src.imports import tf
from src.split.binarysplit import UnitSplit

TRAINING_SET_SIZE = 2000
VALIDATION_SET_SIZE = 100
FEATURE_COUNT = 32
CLASS_COUNT = 20

BATCH_SIZE = 128


# https://patrykchrabaszcz.github.io/Imagenet32/


def foo():
    train, test = tf.keras.datasets.cifar10.load_data()

    train_size = train[0].shape[0]
    print(train_size)

    train_set, validation_set = split_dataset(
        UnitSplit.from_second(1 / 4),
        from_numpy(train)
    )
    print(train_set)
    print(validation_set)


def get_datasets() -> Tuple[Dataset, Dataset]:
    dataset = random_dataset(TRAINING_SET_SIZE).shuffle(
        buffer_size=TRAINING_SET_SIZE,
        reshuffle_each_iteration=True
    ).batch(BATCH_SIZE)

    validation_dataset = random_dataset(VALIDATION_SET_SIZE).batch(BATCH_SIZE)
    return dataset, validation_dataset


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
