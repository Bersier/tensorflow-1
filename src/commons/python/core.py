import datetime
import pickle
from functools import reduce
from operator import mul
from typing import Iterable, Tuple, TypeVar, Dict, Mapping, Union, List, Callable, Any

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')


def on_first(f: Callable[[S], T]) -> Callable[[S, U], Tuple[T, U]]:
    def closure(s: S, u: U) -> Tuple[T, U]:
        return f(s), u

    return closure


def reverse_map(mapping: Iterable[Tuple[S, T]]) -> Dict[T, S]:
    return dict(flip(mapping))


def flip(iterable: Iterable[Tuple[S, T]]) -> Iterable[Tuple[T, S]]:
    return map(lambda t: (t[1], t[0]), iterable)


def to_list(mapping: Iterable[Tuple[S, T]], key_to_index: Mapping[S, int], length: int) -> List[T]:
    r = fill(length, None)
    for k, v in mapping:
        r[key_to_index[k]] = v
    return r


def fill(length: int, value: T) -> List[T]:
    return [value] * length


def product(numbers: Iterable[Union[int, float]]) -> Union[int, float]:
    return reduce(mul, numbers, 1)


def todo(*args):
    raise Exception("Not Implemented. Input:", args)


def conjunction(collection, predicate):
    for e in collection:
        if not predicate(e):
            return False
    return True


def disjunction(collection, predicate):
    for e in collection:
        if predicate(e):
            return True
    return False


def negation(function):
    def n(x):
        return -function(x)

    return n


def pickle_to_file(data, filename):
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()


def unpickle_file(filename):
    file = open(filename, 'rb')
    new = pickle.load(file)
    file.close()
    return new


def present_time():
    now = datetime.datetime.now()
    return now.strftime("%m%d-%H%M%S")


def new_instance(of: type, with_fields: Mapping[str, Any]):
    """Warning! Unsafe."""
    # noinspection PyArgumentList
    obj = of.__new__(of)
    for name, value in with_fields.items():
        setattr(obj, name, value)
    return obj
