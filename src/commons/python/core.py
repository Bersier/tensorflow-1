import pickle
from functools import reduce
from operator import mul
from typing import Iterable, Tuple, TypeVar, Dict, Mapping, Union, List, Callable, Any, Sequence

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')


def pair_1_lifted(f: Callable[[S], T]) -> Callable[[S, U], Tuple[T, U]]:
    """
    :param f:     S -> T
    :return: (S, U) -> (T, U)
    """
    def closure(s: S, u: U) -> Tuple[T, U]:
        return f(s), u

    return closure


def index_map(sequence: Sequence[T]) -> Dict[T, int]:
    """
    A sequence can be seen as a map from int to T.
    This function reverses that map.
    :return: a map from the elements in @sequence to their indices
    """
    return reverse_map(enumerate(sequence))


def reverse_map(mapping: Iterable[Tuple[S, T]]) -> Dict[T, S]:
    """
    Reverses the given map.
    In case of conflict, the last occurrence wins.
    """
    return dict(map(flipped, mapping))


def flipped(p: Tuple[S, T]) -> Tuple[T, S]:
    return p[1], p[0]


def to_list(mapping: Iterable[Tuple[S, T]], key_to_index: Mapping[S, int], length: int) -> List[T]:
    r = filled(length, None)
    for k, v in mapping:
        r[key_to_index[k]] = v
    return r


def filled(length: int, value: T) -> List[T]:
    return [value] * length


def product(numbers: Iterable[Union[int, float]]) -> Union[int, float]:
    return reduce(mul, numbers, 1)


def todo(*args):
    raise Exception("Not Implemented. Input:", args)


def conjunction(elements: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    for element in elements:
        if not predicate(element):
            return False
    return True


def disjunction(elements: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    for element in elements:
        if predicate(element):
            return True
    return False


def negated(function: Callable[[S], T]) -> Callable[[S], T]:
    def n(x):
        return -function(x)

    return n


def pickle_to_file(data: Any, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def unpickle_file(filename: str) -> Any:
    with open(filename, 'rb') as file:
        return pickle.load(file)


def new_instance(of: type[T], with_fields: Mapping[str, Any]) -> T:
    """Warning! Unsafe."""
    # noinspection PyArgumentList
    obj = of.__new__(of)
    for name, value in with_fields.items():
        setattr(obj, name, value)
    return obj
