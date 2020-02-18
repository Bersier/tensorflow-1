from enum import Enum, auto


class Axis(Enum):
    Stock = auto()
    Time = auto()
    Feature = auto()
    SharedFeature = auto()
    Batch = auto()
    Run = auto()
    Example = auto()
    State = auto()
    SharedState = auto()
    T01 = auto()
    T02 = auto()
    T1s = auto()
    Ts2 = auto()
    T34 = auto()
    Output = auto()
    SharedOutput = auto()
