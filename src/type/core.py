from dataclasses import dataclass
from typing import Union, Callable, List, Sequence, Generic, TypeVar

from tensorflow.keras import losses
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from src.commons.imports import tf
from src.commons.python.core import product

SHAPE_TYPE = Sequence[int]

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Weighted(Generic[T]):
    value: T
    weight: float


@dataclass(frozen=True)
class Along:
    tensor: tf.Tensor
    axes: List[int]


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
class SimpleTrainingSpec:
    validation_holdout_fraction: float
    epoch_count: int
    batch_size: int
    model_maker: Callable[[IOType], tf.keras.Model]
    optimizer: Optimizer


@dataclass(frozen=True)
class LearningProblem:
    dataset: SizedDataset
    loss_function: Union[Loss, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]
    metrics: List[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]
    io_type: IOType

    def data(self) -> tf.data.Dataset:
        return self.dataset.data

    @staticmethod
    def with_default_crossentropy(
            dataset: SizedDataset,
            io_type: IOType,
            metrics=None
    ):
        if metrics is None:
            metrics = []
        return LearningProblem(
            dataset=dataset,
            loss_function=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.sparse_categorical_accuracy] + metrics,
            io_type=io_type
        )


@dataclass(frozen=True)
class WholeDatasetSize:
    pass
