from dataclasses import dataclass
from typing import Callable

from tensorflow import Tensor

from src.data.sizeddataset import SizedDataset


@dataclass(frozen=True)
class LearningProblem:
    dataset: SizedDataset
    loss_function: Callable[[Tensor, Tensor], Tensor]
    input_shape: Tensor
    output_shape: Tensor
