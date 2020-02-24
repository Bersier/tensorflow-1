from typing import Union, Optional, List, Callable, Tuple

from src.commons.imports import tf

AXIS_TYPE = Union[int, Optional[List[int]]]
TENSOR_FUNCTION = Callable[[tf.Tensor], tf.Tensor]
TENSOR_PAIR = Tuple[tf.Tensor, tf.Tensor]
