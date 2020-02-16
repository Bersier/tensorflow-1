import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, TypeVar

from src.imports import tf

T = TypeVar('T')
S = TypeVar('S')


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


def split(binary_unit_split: BinaryUnitSplit, dataset: tf.data.Dataset, size: int) \
        -> (tf.data.Dataset, tf.data.Dataset):
    first_size = to_int_split(binary_unit_split, size).first
    return dataset.take(first_size), dataset.skip(first_size)


def approximate(unit_split: Sequence[float], total: int) -> Sequence[int]:
    assert sum(unit_split) == 1
    assert all(map(lambda x: x >= 0, unit_split))
    assert total >= 0

    def score(x: float) -> (float, float):
        if x == 0:
            return 0, 0
        tx = total * x
        floor = math.floor(tx)
        if floor == 0:
            return x, 0
        return 0, x * math.log(math.ceil(tx) / floor)

    def indexed_score(pair: Tuple[int, float]) -> (float, float, int):
        i, x = pair
        s1, s2 = score(x)
        return s1, s2, i

    ordered = sorted(map(indexed_score, enumerate(unit_split)), reverse=True)
    result = list(map(lambda x: int(total * x), unit_split))
    left_over = total - sum(result)
    for _, _, index in ordered:
        if left_over == 0:
            return result
        result[index] += 1
        left_over -= 1
    return result


def flip(iterable: Iterable[Tuple[S, T]]) -> Iterable[Tuple[T, S]]:
    return map(lambda t: (t[1], t[0]), iterable)


def to_int_split(unit_split: BinaryUnitSplit, total: int) -> BinaryIntSplit:
    """
    Convert the given unit split to an integer split of the given total.
    
    It returns the integer split that, when normalized,
    maximizes the mutual information with the given unit split.
    """
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
    first, second = split(BinaryUnitSplit.from_second(1 / 4), dataset, 3)
    for a in first:
        print(a)
    print()
    for a in second:
        print(a)
    print()


def test_approximate():
    print(approximate([0.3, 0.2, 0.1, 0.4], 11))


test_approximate()
# test_split()
# test_to_int_split()
