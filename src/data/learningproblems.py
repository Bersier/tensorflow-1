from src.commons.imports import tf
from src.data.random import random_dataset, DatasetSpec
from src.data.utils import from_numpy, normalized
from src.type.core import LearningProblem, IOType


def cifar10():
    train, _ = tf.keras.datasets.cifar10.load_data()
    xs, ys = train
    return LearningProblem.with_default_crossentropy(
        dataset=from_numpy((normalized(xs), ys)),
        io_type=IOType([32, 32, 3], [10])
    )

# map over dataset to add nans or zeros


def random(spec: DatasetSpec):
    return LearningProblem.with_default_crossentropy(
        dataset=random_dataset(spec),
        io_type=IOType([spec.feature_count], [spec.class_count])
    )

# https://patrykchrabaszcz.github.io/Imagenet32/
