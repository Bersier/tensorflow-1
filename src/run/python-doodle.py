from typing import Tuple, Mapping, Any


def new_instance(of: type, with_fields: Mapping[str, Any]):
    obj = of.__new__(of)
    for attr, value in with_fields.items():
        setattr(obj, attr, value)
    return obj


class A:
    """Example class"""

    def __init__(self, pair: Tuple[int, int]):
        self.first = pair[0]
        self.second = pair[1]

    def sum(self):
        return self.first + self.second


# Example use of new_instance
a_instance = new_instance(
    of=A,
    with_fields={'first': 1, 'second': 2}
)

print(a_instance)
print(a_instance.first)
print(a_instance.second)
