import math
from dataclasses import dataclass

from src.imports import tf


@dataclass(frozen=True)
class BinaryIntSplit:
    first: int
    second: int

    def __post_init__(self):
        assert self.first >= 0
        assert self.second >= 0

    def total(self):
        return self.first + self.second


@dataclass(frozen=True)
class BinaryUnitSplit:
    first: float
    second: float

    def __post_init__(self):
        assert self.first >= 0
        assert self.second >= 0
        assert self.first + self.second == 1

    @staticmethod
    def total():
        return 1

    @staticmethod
    def from_first(first):
        return BinaryUnitSplit(first, 1 - first)

    @staticmethod
    def from_second(second):
        return BinaryUnitSplit(1 - second, second)

    @staticmethod
    def normalized(first, second):
        normalized_first = first / (first + second)
        return BinaryUnitSplit(normalized_first, 1 - normalized_first)


def split(binary_unit_split: BinaryUnitSplit, dataset: tf.data.Dataset, size: int)\
        -> (tf.data.Dataset, tf.data.Dataset):
    first_size = to_int_split(binary_unit_split, size).first
    return dataset.take(first_size), dataset.skip(first_size)


def to_int_split(unit_split: BinaryUnitSplit, total: int) -> BinaryIntSplit:
    """
    Convert the given unit split to an integer split of the given total.
    
    It returns the integer split that, when normalized,
    maximizes the mutual information with the given unit split.
    """
    # -H(u, i) = u.first*log(i.first / total) + u.second*log(i.second / total)
    # -H(u, i) = u1 * log(i1 / t) + u2 * log(i2 / t)

    # u1 * log(floor(u1*t) + 0) + u2 * log(floor(u2*t) + 1)
    # u1 * log(floor(u1*t) + 1) + u2 * log(floor(u2*t) + 0)

    # u1 * log((floor(u1*t) + 1) / floor(u1*t))
    # u2 * log((floor(u2*t) + 1) / floor(u2*t))
    if total == 0:
        return BinaryIntSplit(0, 0)

    if unit_split.first == 0:
        return BinaryIntSplit(0, total)
    if unit_split.second == 0:
        return BinaryIntSplit(total, 0)

    first_part = int(unit_split.first * total)
    second_part = int(unit_split.second * total)

    if first_part == 0:
        return BinaryIntSplit(1, second_part)
    if second_part == 0:
        return BinaryIntSplit(first_part, 1)

    if first_part + second_part == total:
        return BinaryIntSplit(first_part, second_part)

    first_score = unit_split.first * math.log((first_part + 1) / first_part)
    second_score = unit_split.second * math.log((second_part + 1) / second_part)

    # This condition is used to maximize the mutual information
    # between the original split and the returned split.
    if first_score > second_score:
        return BinaryIntSplit(first_part + 1, second_part)
    else:
        return BinaryIntSplit(first_part, second_part + 1)


def test_to_int_split():
    print(to_int_split(BinaryUnitSplit(0.0, 1.0), 5))
    print(to_int_split(BinaryUnitSplit(0.1, 0.9), 5))
    print(to_int_split(BinaryUnitSplit(0.2, 0.8), 5))
    print(to_int_split(BinaryUnitSplit(0.3, 0.7), 5))
    print(to_int_split(BinaryUnitSplit(0.4, 0.6), 5))
    print(to_int_split(BinaryUnitSplit(0.5, 0.5), 5))
    print(to_int_split(BinaryUnitSplit(0.6, 0.4), 5))


def test_split():
    dataset = tf.data.Dataset.range(3)
    first, second = split(BinaryUnitSplit.from_second(1/4), dataset, 3)
    for a in first:
        print(a)
    print()
    for a in second:
        print(a)
    print()


# test_split()
# test_to_int_split()
