from tensorflow.keras import initializers
from tensorflow.keras import layers

from src.commons.imports import tf
from src.model.layer.core import dense_relu_layer
from src.model.layer.doubledense import DoubleDense
from src.model.layer.optioncase import OptionCase
from src.type.core import IOType


def new_option_model(io_type: IOType) -> tf.keras.Model:
    assert len(io_type.output_shape) == 1

    inp = layers.Input(shape=io_type.input_shape)
    x = inp
    x = layers.Flatten()(x)
    x = layers.Reshape([io_type.input_size(), 1])(x)

    x = OptionCase(repr_length=3)(x)

    x = DoubleDense(
        unit_count=64,
        wee_kernel_initializer=initializers.he_normal(),
        big_kernel_initializer=initializers.he_normal()
    )(x)
    x = layers.ReLU()(x)

    x = dense_relu_layer(x, width=32)

    out = layers.Dense(
        io_type.output_size(),
        kernel_initializer=initializers.zeros
    )(x)

    return tf.keras.models.Model(inp, out)
