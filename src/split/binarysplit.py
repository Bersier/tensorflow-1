from dataclasses import dataclass


@dataclass(frozen=True)
class IntSplit:
    first: int
    second: int

    def __post_init__(self):
        assert self.first >= 0
        assert self.second >= 0

    def total(self):
        return self.first + self.second

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.first
        if index == 1:
            return self.second
        raise IndexError()


@dataclass(frozen=True)
class UnitSplit:
    first: float
    second: float

    def __post_init__(self):
        assert self.first >= 0
        assert self.second >= 0
        assert self.first + self.second == 1

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.first
        if index == 1:
            return self.second
        raise IndexError()

    @staticmethod
    def total():
        return 1

    @staticmethod
    def from_first(first):
        return UnitSplit(first, 1 - first)

    @staticmethod
    def from_second(second):
        return UnitSplit(1 - second, second)

    @staticmethod
    def normalized(first, second):
        normalized_first = first / (first + second)
        return UnitSplit(normalized_first, 1 - normalized_first)
