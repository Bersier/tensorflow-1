from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from src.data import FEATURE_COUNT, CLASS_COUNT
from src.imports import tf


def new_model(optimizer: Optimizer, loss: Loss) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(FEATURE_COUNT,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(CLASS_COUNT)])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    return model
