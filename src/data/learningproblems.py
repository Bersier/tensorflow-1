from src.data.utils import from_numpy, normalized
from src.imports import tf

from src.types.classes import LearningProblem


def cifar10():
    train, _ = tf.keras.datasets.cifar10.load_data()
    xs, ys = train
    return LearningProblem(
        dataset=from_numpy((normalized(xs), ys)),
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy],
        input_shape=(32, 32, 3),
        output_shape=(10,)
    )

# https://patrykchrabaszcz.github.io/Imagenet32/
