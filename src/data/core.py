from typing import Tuple, Union

import numpy as np
from numpy import ndarray

from src.commons.imports import AUTOTUNE, tf
from src.split.binarysplit import UnitSplit
from src.split.splitconversion import to_int_split
from src.type.core import SizedDataset, WholeDatasetSize


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
