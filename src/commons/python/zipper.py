from __future__ import annotations

from dataclasses import dataclass
from inspect import ismethod
from typing import Generic, TypeVar, Mapping, Any, cast

from src.commons.python.type import List, Cons

T = TypeVar('T', covariant=True)


@dataclass(frozen=True)
class Node:
    of_type: type
    attribute_name: str
    other_fields: Mapping[str, Any]


@dataclass(frozen=True)
class Zipper(Generic[T]):
    path: List[Node]
    position: T

    def __getattr__(self, name):
        attribute = getattr(self.position, name)
        if ismethod(attribute):
            return attribute
        else:
            return self.descend(name)

    def descend(self, name: str) -> Zipper:
        fields = dict(vars(self.position))
        del fields[name]
        node = Node(type(self.position), name, fields)
        return Zipper(
            path=Cons(node, self.path),
            position=getattr(self.position, name)
        )

    def ascend(self) -> Zipper:
        path = cast(Cons[Node], self.path)
        node = path.head
        fields = dict(node.other_fields)
        fields[node.attribute_name] = self.position
        position = new_instance(node.of_type, fields)
        return Zipper(
            path=path.tail,
            position=position
        )


def new_instance(of: type, with_fields: Mapping[str, Any]):
    # noinspection PyArgumentList
    obj = of.__new__(of)
    for name, value in with_fields.items():
        setattr(obj, name, value)
    return obj
