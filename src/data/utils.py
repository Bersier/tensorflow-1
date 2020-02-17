from typing import Tuple

from numpy import ndarray

from src.data.sizeddataset import SizedDataset, Dataset
from src.imports import tf
from src.split.binarysplit import UnitSplit
from src.split.splitconversion import to_int_split

FRACTION_SET_ASIDE_FOR_VALIDATION = 1 / 4
BATCH_SIZE = 128


def get_dataset_from_numpy(examples: Tuple[ndarray, ndarray]) -> Tuple[SizedDataset, SizedDataset]:
    train_set, validation_set = split_dataset(
        UnitSplit.from_second(FRACTION_SET_ASIDE_FOR_VALIDATION),
        from_numpy(examples)
    )

    print(train_set)
    print(validation_set)

    return ready_for_use(train_set, BATCH_SIZE), ready_for_use(validation_set, BATCH_SIZE)


def from_numpy(numpy_set: Tuple[ndarray, ndarray]) -> SizedDataset:
    return SizedDataset(
        data=Dataset.from_tensor_slices(numpy_set),
        size=numpy_set[0].shape[0]
    )


def ready_for_use(dataset: SizedDataset, batch_size: int, shuffle_buffer_size: int = None) -> SizedDataset:
    if not shuffle_buffer_size:
        shuffle_buffer_size = dataset.size
    data = dataset.data.shuffle(
        buffer_size=shuffle_buffer_size,
        reshuffle_each_iteration=True
    ).batch(
        batch_size
    ).prefetch(
        buffer_size=AUTOTUNE
    )
    return SizedDataset(data, dataset.size)


def split_dataset(binary_unit_split: UnitSplit, dataset: SizedDataset) -> (SizedDataset, SizedDataset):
    data = dataset.data.shuffle(buffer_size=dataset.size, reshuffle_each_iteration=False)
    first_size, second_size = to_int_split(binary_unit_split, dataset.size)
    first_dataset = SizedDataset(data.take(first_size), first_size)
    second_dataset = SizedDataset(data.skip(first_size), second_size)
    return first_dataset, second_dataset


def test_split_dataset():
    dataset_size = 3
    dataset = SizedDataset(Dataset.range(dataset_size), dataset_size)
    first, second = split_dataset(UnitSplit.from_second(1 / 4), dataset)
    for a in first.data:
        print(a)
    print()
    for a in second.data:
        print(a)
    print()


# test_split_dataset()
AUTOTUNE = tf.data.experimental.AUTOTUNE
