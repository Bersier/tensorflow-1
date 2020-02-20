import datetime
import pickle
from functools import reduce
from operator import mul
from typing import Iterable, Tuple, TypeVar, Dict, Mapping, Union, List

T = TypeVar('T')
S = TypeVar('S')


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
    infile = open(filename, 'rb')
    new = pickle.load(infile)
    infile.close()
    return new


def present_time():
    now = datetime.datetime.now()
    return now.strftime("%m%d-%H%M%S")
