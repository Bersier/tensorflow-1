from tensorflow.keras import initializers
from tensorflow.keras import layers

from src.imports import tf
from src.types.classes import ModelSpec


def new_flat_model(model_spec: ModelSpec) -> tf.keras.Model:
    assert len(model_spec.output_shape) == 1

    inp = layers.Input(shape=model_spec.input_shape)
    x = inp
    x = layers.Flatten()(x)

    x = dense_relu_layer(x, width=64)  # 2 * model_spec.input_size())
    x = dense_relu_layer(x, width=32)  # 2 * model_spec.input_size())

    out = layers.Dense(
        model_spec.output_size(),
        kernel_initializer=initializers.zeros
    )(x)

    return tf.keras.models.Model(inp, out)


def dense_relu_layer(inp: tf.Tensor, width: int) -> tf.Tensor:
    x = layers.Dense(width, kernel_initializer=initializers.he_normal())(inp)
    return layers.ReLU()(x)
