from typing import Iterable, Tuple, TypeVar, Dict, Mapping

T = TypeVar('T')
S = TypeVar('S')


def reverse_map(mapping: Iterable[S, T]) -> Dict[T, S]:
    return dict(flip(mapping))


def flip(iterable: Iterable[Tuple[S, T]]) -> Iterable[Tuple[T, S]]:
    return map(lambda t: (t[1], t[0]), iterable)


def to_list(mapping: Iterable[S, T], key_to_index: Mapping[S, int], length: int):
    r = [None] * length
    for k, v in mapping:
        r[key_to_index[k]] = v
    return r
