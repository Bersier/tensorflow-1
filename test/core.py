from src.commons.imports import tf

from src.data.core import split_dataset
from src.split.binarysplit import UnitSplit
from src.type.core import SizedDataset


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
