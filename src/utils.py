from typing import Iterable, Tuple, TypeVar

T = TypeVar('T')
S = TypeVar('S')


def flip(iterable: Iterable[Tuple[S, T]]) -> Iterable[Tuple[T, S]]:
    return map(lambda t: (t[1], t[0]), iterable)
