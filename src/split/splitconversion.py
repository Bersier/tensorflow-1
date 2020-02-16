import math

from src.split.binarysplit import UnitSplit, IntSplit


def to_int_split(unit_split: UnitSplit, total: int) -> IntSplit:
    """
    Convert the given unit split to an integer split of the given total.

    It returns the integer split that, when normalized,
    minimizes the cross entropy with the given unit split.
    """
    if total <= 1:
        if unit_split.first > unit_split.second:
            return IntSplit(total, 0)
        else:
            return IntSplit(0, total)

    first_part = int(unit_split.first * total)
    second_part = int(unit_split.second * total)

    if first_part + second_part == total:
        return IntSplit(first_part, second_part)

    if first_part == 0:
        return IntSplit(1, second_part)
    if second_part == 0:
        return IntSplit(first_part, 1)

    first_score = unit_split.first * math.log((first_part + 1) / first_part)
    second_score = unit_split.second * math.log((second_part + 1) / second_part)

    # This condition is used to minimize the cross entropy
    # between the original split and the returned split.
    if first_score > second_score:
        return IntSplit(first_part + 1, second_part)
    else:
        return IntSplit(first_part, second_part + 1)


def test_to_int_split():
    print(to_int_split(UnitSplit(0.0, 1.0), 5))
    print(to_int_split(UnitSplit(0.1, 0.9), 5))
    print(to_int_split(UnitSplit(0.2, 0.8), 5))
    print(to_int_split(UnitSplit(0.3, 0.7), 5))
    print(to_int_split(UnitSplit(0.4, 0.6), 5))
    print(to_int_split(UnitSplit(0.5, 0.5), 5))
    print(to_int_split(UnitSplit(0.6, 0.4), 5))

# test_to_int_split()
