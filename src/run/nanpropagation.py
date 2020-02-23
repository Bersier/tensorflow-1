import tensorflow as tf

from tensorflow.keras import layers, losses, models

FEATURE_COUNT = 2
TRAINING_SET_SIZE = 128


def patch_nans(t: tf.Tensor) -> tf.Tensor:
    """:return t with nans replaced by zeros"""
    nan_mask = tf.math.is_nan(t)
    return tf.where(nan_mask, tf.zeros_like(t), t)


def check_numerics(t: tf.Tensor) -> tf.Tensor:
    """Throw an exception if t contains nans."""
    return tf.debugging.check_numerics(t, "t")


def get_model() -> models.Model:
    inp = layers.Input(shape=[FEATURE_COUNT])

    # Hidden layer
    mid = layers.Dense(units=64)(inp)
    mid = layers.ReLU()(mid)

    mid = layers.Dense(units=1)(mid)

    mid = layers.Lambda(patch_nans)(mid)
    out = layers.Lambda(check_numerics)(mid)

    return models.Model(inp, out)


model = get_model()
model.compile(
    optimizer=tf.optimizers.SGD(),
    loss=losses.mean_squared_error
)
model.summary()

features = tf.random.normal(shape=[TRAINING_SET_SIZE, FEATURE_COUNT])
features_with_nans = tf.maximum(tf.math.log(features + 1), tf.zeros_like(features))
tf.print(-features_with_nans)
tf.print(tf.zeros(shape=[2]) / tf.zeros(shape=[2]))
labels = tf.random.normal(shape=[TRAINING_SET_SIZE, 1])

# Evaluate the model before training
model.evaluate(features_with_nans, labels, batch_size=8)

# Evaluate the model while training
model.fit(features_with_nans, labels, batch_size=8, epochs=4)
