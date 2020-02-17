import math
from typing import Sequence, Tuple


def approximate(unit_split: Sequence[float], total: int) -> Sequence[int]:
    """
    Convert the given unit split to an integer split of the given total.

    It returns the integer split that, when normalized,
    minimizes the cross entropy with the given unit split,
    while satisfying the constraint that each chunk corresponds to either
    the floor or the ceiling of the product of its size and the total.
    """
    assert sum(unit_split) == 1
    assert all(map(lambda x: x >= 0, unit_split))
    assert total >= 0

    def score(x: float) -> (float, float):
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


def test_approximate():
    print(approximate([0.3, 0.2, 0.1, 0.01, 0.02, 0.03, 0.4 - 0.06], 11))

# test_approximate()
