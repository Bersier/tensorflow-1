from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import ismethod
from typing import Generic, TypeVar, Mapping, Any, cast

from src.commons.python.core import new_instance
from src.commons.python.list import List, Cons

T = TypeVar('T', covariant=True)


@dataclass(frozen=True)
class Node(Generic[T]):
    of_type: type[T]
    attribute_name: str
    other_fields: Mapping[str, Any]
    # todo why isn't the original object saved here instead?


@dataclass(frozen=True)
class Zipper(Generic[T]):
    path: List[Node]
    focus: T

    def __getattr__(self, name):
        attribute = getattr(self.focus, name)
        if ismethod(attribute):
            return attribute
        else:
            return self.at(name)

    def with_updates(self, **changes) -> Zipper[T]:
        return Zipper(
            path=self.path,
            focus=replace(self.focus, **changes)
        )

    def at(self, name: str) -> Zipper:
        fields = dict(vars(self.focus))
        del fields[name]
        node = Node(
            of_type=type(self.focus),
            attribute_name=name,
            other_fields=fields,
        )
        return Zipper(
            path=Cons(node, self.path),
            focus=getattr(self.focus, name)
        )

    def up(self) -> Zipper:
        path = cast(Cons[Node], self.path)
        node = path.head
        fields = dict(node.other_fields)
        fields[node.attribute_name] = self.focus
        position = new_instance(node.of_type, fields)
        return Zipper(
            path=path.tail,
            focus=position
        )

    def root(self) -> Any:
        if self.path == List.empty():
            return self.focus
        else:
            return self.up().root()


def zipper_of(value: T) -> Zipper[T]:
    return Zipper(List.empty(), value)
