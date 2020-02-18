from functools import reduce
from operator import mul
from typing import Iterable, Tuple, TypeVar, Dict, Mapping, Union

T = TypeVar('T')
S = TypeVar('S')


def reverse_map(mapping: Iterable[Tuple[S, T]]) -> Dict[T, S]:
    return dict(flip(mapping))


def flip(iterable: Iterable[Tuple[S, T]]) -> Iterable[Tuple[T, S]]:
    return map(lambda t: (t[1], t[0]), iterable)


def to_list(mapping: Iterable[Tuple[S, T]], key_to_index: Mapping[S, int], length: int):
    r = [None] * length
    for k, v in mapping:
        r[key_to_index[k]] = v
    return r


def product(numbers: Iterable[Union[int, float]]) -> Union[int, float]:
    return reduce(mul, numbers, 1)


def todo(*args):
    raise Exception("Not Implemented. Input:", args)
