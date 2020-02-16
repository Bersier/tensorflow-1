from tensorflow.data import Dataset

from src.data.sizeddataset import SizedDataset
from src.split.binarysplit import UnitSplit
from src.split.splitconversion import to_int_split


def split_dataset(binary_unit_split: UnitSplit, dataset: SizedDataset) -> (SizedDataset, SizedDataset):
    dataset = dataset.data.shuffle(dataset.size, reshuffle_each_iteration=False)
    first_size, second_size = to_int_split(binary_unit_split, dataset.size)
    first_dataset = SizedDataset(dataset.take(first_size), first_size)
    second_dataset = SizedDataset(dataset.skip(first_size), second_size)
    return first_dataset, second_dataset


def test_split():
    dataset_size = 3
    dataset = SizedDataset(Dataset.range(dataset_size), dataset_size)
    first, second = split_dataset(UnitSplit.from_second(1 / 4), dataset)
    for a in first.data:
        print(a)
    print()
    for a in second.data:
        print(a)
    print()


# test_split()
