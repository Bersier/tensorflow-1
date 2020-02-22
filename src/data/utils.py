from typing import Tuple, Union

import numpy as np
from numpy import ndarray

from src.commons.imports import tf
from src.split.binarysplit import UnitSplit
from src.split.splitconversion import to_int_split
from src.type.core import WholeDatasetSize, SizedDataset

FRACTION_SET_ASIDE_FOR_VALIDATION = 1 / 4
BATCH_SIZE = 4


def dataset_from_numpy(examples: Tuple[ndarray, ndarray]) -> Tuple[SizedDataset, SizedDataset]:
    train_set, validation_set = split_dataset(
        UnitSplit.from_second(FRACTION_SET_ASIDE_FOR_VALIDATION),
        from_numpy(examples)
    )

    return ready_for_training(train_set, BATCH_SIZE), ready_for_evaluation(validation_set, BATCH_SIZE)


def from_numpy(numpy_dataset: Tuple[ndarray, ndarray]) -> SizedDataset:
    return SizedDataset(
        data=tf.data.Dataset.from_tensor_slices(numpy_dataset),
        size=numpy_dataset[0].shape[0]
    )


def normalized(xs: ndarray, sample_axis: int = 0) -> ndarray:
    xs_mean = np.mean(xs, axis=sample_axis)
    xs_std = np.std(xs, axis=sample_axis)
    return (xs - xs_mean) / xs_std


def ready_for_training(
        dataset: SizedDataset,
        batch_size: int,
        shuffle_buffer_size: Union[int, WholeDatasetSize] = WholeDatasetSize
) -> SizedDataset:
    if shuffle_buffer_size == WholeDatasetSize:
        shuffle_buffer_size = dataset.size
    data = dataset.data.shuffle(
        buffer_size=shuffle_buffer_size,
        reshuffle_each_iteration=True
    )
    return ready_for_evaluation(SizedDataset(data, dataset.size), batch_size)


def ready_for_evaluation(dataset: SizedDataset, batch_size: int) -> SizedDataset:
    data = dataset.data.batch(
        batch_size
    ).prefetch(
        buffer_size=AUTOTUNE
    )
    return SizedDataset(data, dataset.size)


def split_dataset(binary_split: UnitSplit, dataset: SizedDataset) -> (SizedDataset, SizedDataset):
    data = dataset.data.shuffle(buffer_size=dataset.size, reshuffle_each_iteration=False)
    first_size, second_size = to_int_split(binary_split, dataset.size)
    first_dataset = SizedDataset(data.take(first_size), first_size)
    second_dataset = SizedDataset(data.skip(first_size), second_size)
    return first_dataset, second_dataset


def test_split_dataset():
    dataset_size = 3
    dataset = SizedDataset(tf.data.Dataset.range(dataset_size), dataset_size)
    first, second = split_dataset(UnitSplit.from_second(1 / 4), dataset)
    for a in first.data:
        print(a)
    print()
    for a in second.data:
        print(a)
    print()


# test_split_dataset()
AUTOTUNE = tf.data.experimental.AUTOTUNE
