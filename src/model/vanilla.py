from tensorflow.keras import initializers
from tensorflow.keras import layers

from src.commons.imports import tf
from src.model.utils import dense_relu_layer
from src.types.classes import IOType


def new_flat_model(io_type: IOType) -> tf.keras.Model:
    assert len(io_type.output_shape) == 1

    inp = layers.Input(shape=io_type.input_shape)
    x = inp
    x = layers.Flatten()(x)

    x = dense_relu_layer(x, width=64)  # 2 * model_spec.input_size())
    x = dense_relu_layer(x, width=32)  # model_spec.input_size())

    out = layers.Dense(
        io_type.output_size(),
        kernel_initializer=initializers.zeros
    )(x)

    return tf.keras.models.Model(inp, out)
