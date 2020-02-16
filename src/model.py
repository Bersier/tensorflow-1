from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from src.data.data import FEATURE_COUNT, CLASS_COUNT
from src.imports import tf


def new_model(optimizer: Optimizer, loss: Loss) -> tf.keras.Model:
    inp = layers.Input(shape=(FEATURE_COUNT,))
    x = inp

    x = dense_relu_layer(x, width=64)
    x = dense_relu_layer(x, width=64)

    out = layers.Dense(CLASS_COUNT, kernel_initializer=initializers.zeros)(x)

    model = tf.keras.models.Model(inp, out)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    return model


def dense_relu_layer(inp: tf.Tensor, width: int) -> tf.Tensor:
    x = layers.Dense(width, kernel_initializer=initializers.he_normal())(inp)
    return layers.ReLU()(x)
