from tensorflow.keras import layers

from src.imports import tf


def new_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')])

    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    return model
