from dataclasses import dataclass
from typing import Union, Callable, Tuple, List

from tensorflow.keras.losses import Loss
from tensorflow_core import Tensor

from src.imports import tf
from src.utils import product

SHAPE_TYPE = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]


@dataclass(frozen=True)
class SizedDataset:
    data: tf.data.Dataset
    size: int


@dataclass(frozen=True)
class IOType:
    input_shape: SHAPE_TYPE
    output_shape: SHAPE_TYPE

    def input_size(self):
        return product(self.input_shape)

    def output_size(self):
        return product(self.output_shape)


@dataclass(frozen=True)
class LearningProblem:
    dataset: SizedDataset
    loss_function: Union[Loss, Callable[[Tensor, Tensor], Tensor]]
    metrics: List[Callable[[Tensor, Tensor], Tensor]]
    io_type: IOType

    def data(self) -> tf.data.Dataset:
        return self.dataset.data


@dataclass(frozen=True)
class WholeDatasetSize:
    pass
